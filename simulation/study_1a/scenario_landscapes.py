"""
Study 1A — Per-scenario posterior energy landscape (target distribution).

목적
----
Study 1A 4 시나리오 (Easy / Moderate / Difficult / Sparse) 의 target
posterior 분포를 1D mixture energy 형태로 시각화한다. 각 시나리오의
(mu_S, mu_W, mu_N, sigma_theta, alpha, rho) 조합이 만드는 에너지 지형을
한눈에 비교하기 위함.

분포 정의 (study_1b/local_trap_landscape.py 와 동일 형태)
--------
3-mode mixture posterior:
  E(theta) = -log[ rho_S exp(-E_S(theta))
                 + rho_W exp(-E_W(theta))
                 + rho_N exp(-E_N(theta)) ]
- S, W (Y=1):  -log[(1-alpha) sigmoid(theta)] + (theta - mu)^2 / (2 sigma^2)
- N    (Y=0):  -log[1 - (1-alpha) sigmoid(theta)] + (theta - mu)^2 / (2 sigma^2)

CLAUDE.md 절대 금지 사항 준수: 가중합 대신 log-sum-exp mixture 사용.

시나리오 파라미터는 langevin_methods_comparison.py 의 SCENARIOS 와 일치한다.

출력
----
- scenario_landscape_<Name>.png  (시나리오별 단일 패널)
- scenario_landscape_all.png     (4 시나리오 2x2 비교 패널)
- scenario_landscapes.npz        (theta_grid + 각 시나리오 E_grid)

실행
----
python3 simulation/study_1a/scenario_landscapes.py
"""
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import argrelmin, argrelmax


# ── 한글 폰트 ──
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────────────────────────────────
# 시나리오 정의 (langevin_methods_comparison.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "name": "Easy",
        "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
        "mu_S": 2.5, "mu_W": 1.0, "mu_N": -2.5,
        "sigma_theta": 0.35, "alpha": 0.20,
    },
    {
        "name": "Moderate",
        "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
        "mu_S": 2.0, "mu_W": 0.5, "mu_N": -1.8,
        "sigma_theta": 0.50, "alpha": 0.35,
    },
    {
        "name": "Difficult",
        "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
        "mu_S": 1.5, "mu_W": 0.0, "mu_N": -1.0,
        "sigma_theta": 0.60, "alpha": 0.50,
    },
    {
        "name": "Sparse",
        "rho_S": 0.10, "rho_W": 0.18, "rho_N": 0.72,
        "mu_S": 2.0, "mu_W": 1.0, "mu_N": -1.0,
        "sigma_theta": 0.55, "alpha": 0.40,
    },
]


# CLAUDE.md 색상 규칙: S=파랑, W=주황, N=회색
C_S = "#2F6DB2"
C_W = "#D85A30"
C_N = "#6B6B6B"
C_CURVE = "#1D9E75"


# ──────────────────────────────────────────────────────────────────────────
# Energy (1D mixture posterior)
# ──────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def E_Y1(theta, alpha, mu, sigma2):
    return (-np.log(np.clip((1 - alpha) * sigmoid(theta), 1e-10, None))
            + (theta - mu) ** 2 / (2 * sigma2))


def E_Y0(theta, alpha, mu, sigma2):
    return (-np.log(np.clip(1 - (1 - alpha) * sigmoid(theta), 1e-10, None))
            + (theta - mu) ** 2 / (2 * sigma2))


def E_mix(theta, params):
    sigma2 = params["sigma_theta"] ** 2
    ES = E_Y1(theta, params["alpha"], params["mu_S"], sigma2)
    EW = E_Y1(theta, params["alpha"], params["mu_W"], sigma2)
    EN = E_Y0(theta, params["alpha"], params["mu_N"], sigma2)
    log_stack = np.stack([
        -ES + np.log(params["rho_S"]),
        -EW + np.log(params["rho_W"]),
        -EN + np.log(params["rho_N"]),
    ], axis=0)
    log_max = log_stack.max(axis=0)
    return -(log_max + np.log(np.exp(log_stack - log_max).sum(axis=0)))


def find_local_minima(theta, energy):
    idx = argrelmin(energy)[0]
    return [(float(theta[i]), float(energy[i])) for i in idx]


def find_barriers(theta, energy, minima):
    idx = argrelmax(energy)[0]
    barriers = []
    for i in idx:
        x_p, y_p = float(theta[i]), float(energy[i])
        left = [m for m in minima if m[0] < x_p]
        right = [m for m in minima if m[0] > x_p]
        if not left or not right:
            continue
        m_left = max(left, key=lambda m: m[0])
        m_right = min(right, key=lambda m: m[0])
        barriers.append({
            "peak_theta": x_p, "peak_E": y_p,
            "left_min": m_left, "right_min": m_right,
            "dE_left": y_p - m_left[1],
            "dE_right": y_p - m_right[1],
        })
    return barriers


# ──────────────────────────────────────────────────────────────────────────
# 그리기
# ──────────────────────────────────────────────────────────────────────────
def draw_landscape(ax, theta_grid, E_grid, params, scen_name, show_xlabel=True):
    minima = find_local_minima(theta_grid, E_grid)
    maxima_idx = argrelmax(E_grid)[0]
    peak_E = (float(E_grid[maxima_idx].max())
              if len(maxima_idx) > 0 else float(E_grid.max()))
    y_lim_high = peak_E + 1.5
    if minima:
        y_lim_low = max(min(m[1] for m in minima) - 1.0, 0)
    else:
        y_lim_low = max(float(E_grid.min()) - 0.5, 0)

    E_show = np.where(E_grid > y_lim_high, np.nan, E_grid)
    ax.plot(theta_grid, E_show, color=C_CURVE, lw=2.3, zorder=3)

    for mu_key, color in (("mu_N", C_N), ("mu_W", C_W), ("mu_S", C_S)):
        mu_val = params[mu_key]
        ax.axvspan(mu_val - 0.8, mu_val + 0.8, alpha=0.10, color=color, zorder=0)

    mu_to_color = {"N": (params["mu_N"], C_N),
                   "W": (params["mu_W"], C_W),
                   "S": (params["mu_S"], C_S)}
    for label, (mu_val, color_g) in mu_to_color.items():
        if minima:
            closest = min(minima, key=lambda m: abs(m[0] - mu_val))
            if abs(closest[0] - mu_val) <= 1.5:
                x_min, y_min = closest
                ax.plot(x_min, y_min, marker="v", color=color_g,
                        markersize=10, zorder=6)
                ax.axvline(x=x_min, color=color_g, ls="--", lw=1.0, alpha=0.55)
                ax.text(x_min, y_min - 0.5,
                        f"$\\theta$={x_min:.2f}",
                        fontsize=8, ha="center", va="top",
                        color=color_g, fontweight="bold")
        ax.text(mu_val, y_lim_high * 0.92, label, color=color_g, fontsize=12,
                ha="center", fontweight="bold", zorder=5)

    ax.set_xlim(-4, 4)
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.set_xticks(np.arange(-4, 5, 1))
    if show_xlabel:
        ax.set_xlabel(r"$\theta$", fontsize=11)
    ax.set_ylabel(r"$-\log\, p(\theta \mid Y, \alpha)$", fontsize=10)
    ax.grid(True, alpha=0.15)
    ax.set_title(
        f"{scen_name}  "
        rf"|  $\mu$=({params['mu_S']}, {params['mu_W']}, {params['mu_N']}), "
        rf"$\sigma_\theta$={params['sigma_theta']}, $\alpha$={params['alpha']}, "
        rf"$\rho$=({params['rho_S']}, {params['rho_W']}, {params['rho_N']})",
        fontsize=9.5, fontweight="bold", pad=8,
    )
    return minima


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    theta_grid = np.linspace(-4, 4, 1000)
    E_per_scenario = {}

    # ── 시나리오별 단일 패널 PNG ──
    for sc in SCENARIOS:
        E_grid = E_mix(theta_grid, sc)
        E_per_scenario[sc["name"]] = E_grid

        fig, ax = plt.subplots(figsize=(9, 5.5))
        minima = draw_landscape(ax, theta_grid, E_grid, sc, sc["name"])
        fig.text(
            0.5, 0.02,
            "▼ = local minimum  |  음영 = S/W/N basin  |  log-sum-exp mixture",
            ha="center", fontsize=9, color="gray",
        )
        fig.subplots_adjust(left=0.08, right=0.97, top=0.90, bottom=0.13)
        png_path = os.path.join(out_dir, f"scenario_landscape_{sc['name']}.png")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved -> {png_path}")

        barriers = find_barriers(theta_grid, E_grid, minima)
        print(f"[{sc['name']}] local minima:")
        for x, y in minima:
            print(f"   theta={x:+.3f}, E={y:.3f}")
        print(f"[{sc['name']}] inter-basin barriers:")
        for b in barriers:
            print(f"   peak theta={b['peak_theta']:+.3f}, E={b['peak_E']:.3f}  "
                  f"|  dE_left={b['dE_left']:.3f}  dE_right={b['dE_right']:.3f}")
        print()

    # ── 4 시나리오 2x2 비교 패널 ──
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, sc in zip(axes.flat, SCENARIOS):
        draw_landscape(ax, theta_grid, E_per_scenario[sc["name"]], sc,
                       sc["name"], show_xlabel=True)
    fig.suptitle(
        "Study 1A — Per-scenario posterior energy landscape (3-mode mixture)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.text(
        0.5, 0.005,
        "▼ = local minimum  |  음영 = S/W/N basin  |  log-sum-exp mixture posterior",
        ha="center", fontsize=9, color="gray",
    )
    fig.subplots_adjust(left=0.06, right=0.98, top=0.93, bottom=0.06,
                        hspace=0.35, wspace=0.22)
    combined_path = os.path.join(out_dir, "scenario_landscape_all.png")
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {combined_path}")

    # ── npz 저장 ──
    npz_path = os.path.join(out_dir, "scenario_landscapes.npz")
    npz_kwargs = {"theta_grid": theta_grid}
    for sc in SCENARIOS:
        npz_kwargs[f"E_{sc['name']}"] = E_per_scenario[sc["name"]]
        for k, v in sc.items():
            if k == "name":
                continue
            npz_kwargs[f"{sc['name']}_{k}"] = v
    np.savez(npz_path, **npz_kwargs)
    print(f"Saved -> {npz_path}")


if __name__ == "__main__":
    main()
