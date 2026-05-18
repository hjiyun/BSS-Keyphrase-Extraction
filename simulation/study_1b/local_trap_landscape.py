"""
Study 1B — Local trap posterior landscape (target distribution).

목적
----
"AWSGLD 가 local trap 을 실제로 탈출하는가?" 실험의 대상 분포를 정의하고
시각화한다. 여기서는 단일 1D mixture posterior 만 만든다 (시나리오 분기 없음).

분포 정의
--------
3-mode mixture posterior:
  E(theta) = -log[ rho_S exp(-E_S(theta))
                 + rho_W exp(-E_W(theta))
                 + rho_N exp(-E_N(theta)) ]

- S (strong keyphrase, Y=1):  mu_S = +2.5
- W (weak keyphrase, Y=1):    mu_W = +1.0   ← 가운데 얕은 trap
- N (non-keyphrase, Y=0):     mu_N = -1.5   ← 가장 무거운 mass (rho_N=0.6)
- alpha=0.20, sigma_theta=0.35

→ N/W/S 세 basin 사이 분명한 에너지 장벽이 존재. N basin 에서 출발한 chain 이
   W/S 로 넘어가는지 보는 trap-escape 실험의 타깃 분포로 적합하다.

장벽 높이 설계
-------------
mu_N 을 -2.5 (Study_1A Easy 원본) → -1.5 로 당겨 N-W 장벽을 약 13 → 약 7 로
조정. 이는 다음을 만족시키기 위함:
- SGLD (T=1) 의 escape time ~ exp(7) ≈ 1100 step → 5000 iter 안에 거의 stuck.
- AWSGLD 의 adaptive weighting 으로는 비교적 짧은 시간에 escape 가능.
- W (+1.0), S (+2.5) 두 mode 가 명확히 보존 (sigma_theta 는 그대로 유지).

CLAUDE.md 절대 금지 사항 준수
- Posterior energy 를 직접 가중합으로 계산하지 않고 log-sum-exp mixture 사용.

출력
----
- local_trap_landscape.npz : theta_grid, E_grid, minima(theta/E), 파라미터
- local_trap_landscape.png : 단일 패널 시각화 (x=theta, y=Energy)

실행
----
python3 simulation/study_1b/local_trap_landscape.py
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
# 분포 파라미터 (Study_1A의 ControlledEasy_v2 와 동일)
# ──────────────────────────────────────────────────────────────────────────
PARAMS = {
    "mu_S": 2.5, "mu_W": 1.0, "mu_N": -0.8,
    "alpha": 0.20,
    "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
    # mu_N=-0.8: N basin 의 깊이를 약간 더 얕게 해서 sampler 가 N 으로 돌아오는
    # 경로의 장벽을 낮춤 (이전 mu_N=-1.0 에서 N drift 가 너무 심했음).
    "sigma_theta": 0.26,
}


# ──────────────────────────────────────────────────────────────────────────
# Energy (1D mixture posterior)
# ──────────────────────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -700, 700)))


def E_Y1(theta, alpha, mu, sigma2):
    """Y=1 group (S, W) energy: -log[(1-alpha) sigmoid(theta)] + Gaussian prior."""
    return (-np.log(np.clip((1 - alpha) * sigmoid(theta), 1e-10, None))
            + (theta - mu) ** 2 / (2 * sigma2))


def E_Y0(theta, alpha, mu, sigma2):
    """Y=0 group (N) energy: -log[1 - (1-alpha) sigmoid(theta)] + Gaussian prior."""
    return (-np.log(np.clip(1 - (1 - alpha) * sigmoid(theta), 1e-10, None))
            + (theta - mu) ** 2 / (2 * sigma2))


def E_mix(theta, alpha, mu_S, mu_W, mu_N, rho_S, rho_W, rho_N, sigma2):
    """-log mixture posterior. log-sum-exp 트릭으로 수치 안정 계산."""
    ES = E_Y1(theta, alpha, mu_S, sigma2)
    EW = E_Y1(theta, alpha, mu_W, sigma2)
    EN = E_Y0(theta, alpha, mu_N, sigma2)
    log_stack = np.stack([-ES + np.log(rho_S),
                          -EW + np.log(rho_W),
                          -EN + np.log(rho_N)], axis=0)
    log_max = log_stack.max(axis=0)
    return -(log_max + np.log(np.exp(log_stack - log_max).sum(axis=0)))


def find_local_minima(theta, energy):
    """boundary 를 제외한 모든 local minimum (theta, E) 반환."""
    idx = argrelmin(energy)[0]
    return [(float(theta[i]), float(energy[i])) for i in idx]


# ──────────────────────────────────────────────────────────────────────────
# 계산 + 시각화
# ──────────────────────────────────────────────────────────────────────────
def main():
    sigma2 = PARAMS["sigma_theta"] ** 2
    theta_grid = np.linspace(-4, 4, 1000)
    E_grid = E_mix(theta_grid, PARAMS["alpha"],
                   PARAMS["mu_S"], PARAMS["mu_W"], PARAMS["mu_N"],
                   PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"],
                   sigma2)
    minima = find_local_minima(theta_grid, E_grid)
    # inter-basin 장벽 정점 기준으로 y축 상한 산정 (boundary 의 prior 폭증 부분은
    # NaN 처리해서 좌측 평탄 plateau 가 생기지 않도록 함)
    maxima_idx = argrelmax(E_grid)[0]
    peak_E = (float(E_grid[maxima_idx].max())
              if len(maxima_idx) > 0 else float(E_grid.max()))

    # CLAUDE.md 색상 규칙: S=파랑, W=주황, N=회색
    C_S = "#2F6DB2"
    C_W = "#D85A30"
    C_N = "#6B6B6B"

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # y_lim 위로 새는 segment 는 NaN 으로 끊어서 그리지 않음 (clipping plateau 방지)
    y_lim_high = peak_E + 1.5
    E_show = np.where(E_grid > y_lim_high, np.nan, E_grid)
    ax.plot(theta_grid, E_show, color="#1D9E75", lw=2.5, zorder=3)

    # 배경 음영 (S/W/N basin)
    ax.axvspan(PARAMS["mu_N"] - 0.8, PARAMS["mu_N"] + 0.8,
               alpha=0.10, color=C_N, zorder=0)
    ax.axvspan(PARAMS["mu_W"] - 0.8, PARAMS["mu_W"] + 0.8,
               alpha=0.10, color=C_W, zorder=0)
    ax.axvspan(PARAMS["mu_S"] - 0.8, PARAMS["mu_S"] + 0.8,
               alpha=0.10, color=C_S, zorder=0)

    # local minimum marker (각 mu_g 근처에서 가장 가까운 local min)
    mu_to_color = {"N": (PARAMS["mu_N"], C_N),
                   "W": (PARAMS["mu_W"], C_W),
                   "S": (PARAMS["mu_S"], C_S)}
    y_lim_low = max(min(m[1] for m in minima) - 1.0, 0)
    # y_lim_high 는 위에서 peak_E + 1.5 로 이미 계산됨

    for label, (mu_val, color_g) in mu_to_color.items():
        closest = min(minima, key=lambda m: abs(m[0] - mu_val))
        if abs(closest[0] - mu_val) <= 1.5:
            x_min, y_min = closest
            ax.plot(x_min, y_min, marker="v", color=color_g,
                    markersize=11, zorder=6)
            ax.axvline(x=x_min, color=color_g, ls="--", lw=1.2, alpha=0.6)
            ax.text(x_min, y_min - 0.6,
                    f"$\\theta$={x_min:.2f}",
                    fontsize=9, ha="center", va="top",
                    color=color_g, fontweight="bold")
        ax.text(mu_val, y_lim_high * 0.92, label, color=color_g, fontsize=13,
                ha="center", fontweight="bold", zorder=5)

    ax.set_xlim(-4, 4)
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.set_xticks(np.arange(-4, 5, 1))
    ax.set_xlabel(r"$\theta$", fontsize=12)
    ax.set_ylabel(r"$-\log\, p(\theta \mid Y, \alpha)$  [Energy]", fontsize=11)
    ax.grid(True, alpha=0.15)

    ax.set_title(
        "Study 1B Target: 3-mode mixture posterior with local traps\n"
        rf"$\mu_S$={PARAMS['mu_S']}, $\mu_W$={PARAMS['mu_W']}, $\mu_N$={PARAMS['mu_N']}  "
        rf"|  $\alpha$={PARAMS['alpha']}, $\sigma_\theta$={PARAMS['sigma_theta']}  "
        rf"|  $\rho$=({PARAMS['rho_S']}, {PARAMS['rho_W']}, {PARAMS['rho_N']})",
        fontsize=10, fontweight="bold", pad=10,
    )

    fig.text(
        0.5, 0.02,
        "▼ = local minimum  |  음영 = S/W/N basin  |  세 mode 사이 분명한 에너지 장벽 존재",
        ha="center", fontsize=9, color="gray",
    )
    fig.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.13)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(out_dir, "local_trap_landscape.png")
    npz_path = os.path.join(out_dir, "local_trap_landscape.npz")

    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    np.savez(npz_path,
             theta_grid=theta_grid,
             E_grid=E_grid,
             minima_theta=np.array([m[0] for m in minima]),
             minima_E=np.array([m[1] for m in minima]),
             **{f"param_{k}": v for k, v in PARAMS.items()})

    print(f"Saved -> {png_path}")
    print(f"Saved -> {npz_path}")
    print()
    print("Detected local minima (theta, E):")
    for x, y in minima:
        print(f"  theta={x:+.3f}, E={y:.3f}")
    print()
    print("Basin assignment:")
    for label, (mu_val, _) in mu_to_color.items():
        closest = min(minima, key=lambda m: abs(m[0] - mu_val))
        print(f"  {label} basin near mu_{label}={mu_val:+.2f}: "
              f"theta_min={closest[0]:+.3f}, E_min={closest[1]:.3f}")
    print()
    # local maxima 사이의 inter-basin 장벽 높이를 mode 별로 보고
    maxima_idx = argrelmax(E_grid)[0]
    barriers = [(float(theta_grid[i]), float(E_grid[i])) for i in maxima_idx]
    print("Inter-basin barriers (peak between adjacent local minima):")
    for x_p, y_p in barriers:
        # 이 peak 양 옆 가장 가까운 local min 두 개 찾기
        left = [m for m in minima if m[0] < x_p]
        right = [m for m in minima if m[0] > x_p]
        if not left or not right:
            continue
        m_left = max(left, key=lambda m: m[0])
        m_right = min(right, key=lambda m: m[0])
        dE_left = y_p - m_left[1]
        dE_right = y_p - m_right[1]
        print(f"  peak at theta={x_p:+.3f}, E={y_p:.3f}  "
              f"|  barrier from left  (theta={m_left[0]:+.2f}, E={m_left[1]:.2f}) = {dE_left:.3f}  "
              f"|  barrier from right (theta={m_right[0]:+.2f}, E={m_right[1]:.2f}) = {dE_right:.3f}")


if __name__ == "__main__":
    main()
