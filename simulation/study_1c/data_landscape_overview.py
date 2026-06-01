"""
Study 1C — 데이터 분포 + target landscape 시각화 (n=200, n=1500 두 scale).

생성된 data_n{N}_seed0.npz 를 읽어 truth 분포 + 1D mixture posterior energy
landscape 를 한 figure 에 보여준다.

Setup (data_generator.py 기준)
- μ_S=+2.5, μ_W=+1.0, μ_N=-1.5,  α=0.20
- σ_θ : n=200 → 0.20,  n=1500 → 0.26
- damping=0.85,  conflict flip_S_to_0=0.10, flip_N_to_1=0.05
"""
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.signal import argrelmin, argrelmax


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_1B = os.path.join(os.path.dirname(_THIS_DIR), "study_1b")
for _p in (_THIS_DIR, STUDY_1B):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from local_trap_landscape import E_mix  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


C_S, C_W, C_N = "#2F6DB2", "#D85A30", "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}

# Study 1C 의 PARAMS (data_generator.py 와 일치)
PARAMS_BASE = {
    "mu_S": 2.5, "mu_W": 1.0, "mu_N": -1.5,
    "alpha": 0.20,
    "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
}
SCALE_SIGMA = {200: 0.20, 1500: 0.26}


def draw_landscape(ax, sigma_theta, title):
    sigma2 = sigma_theta ** 2
    theta_grid = np.linspace(-3.5, 4.0, 1200)
    E_grid = E_mix(theta_grid, PARAMS_BASE["alpha"],
                   PARAMS_BASE["mu_S"], PARAMS_BASE["mu_W"],
                   PARAMS_BASE["mu_N"],
                   PARAMS_BASE["rho_S"], PARAMS_BASE["rho_W"],
                   PARAMS_BASE["rho_N"], sigma2)

    min_idx = argrelmin(E_grid)[0]
    max_idx = argrelmax(E_grid)[0]
    minima = [(float(theta_grid[i]), float(E_grid[i])) for i in min_idx]
    peak_E = (max(E_grid[i] for i in max_idx) if len(max_idx) > 0
              else float(E_grid.max()))
    y_lim_high = peak_E + 1.5
    y_lim_low = max((min(p[1] for p in minima) if minima else 0.0) - 0.5, 0.0)

    E_show = np.where(E_grid > y_lim_high, np.nan, E_grid)
    ax.plot(theta_grid, E_show, color="#1D9E75", lw=2.4, zorder=3)
    for g_label, mu_val, color in [("S", PARAMS_BASE["mu_S"], C_S),
                                    ("W", PARAMS_BASE["mu_W"], C_W),
                                    ("N", PARAMS_BASE["mu_N"], C_N)]:
        ax.axvline(mu_val, color=color, ls="--", lw=1.3, alpha=0.85)
        ax.axvspan(mu_val - 0.6, mu_val + 0.6, alpha=0.08, color=color)
        ax.text(mu_val, y_lim_high * 0.94, g_label, ha="center",
                color=color, fontsize=12, fontweight="bold")
    for x_m, y_m in minima:
        ax.plot(x_m, y_m, marker="v", color="#1D9E75", markersize=10,
                markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        ax.text(x_m, y_m - 0.3, f"θ={x_m:+.2f}", fontsize=8,
                ha="center", va="top", color="#1D9E75")

    ax.set_xlim(theta_grid.min(), theta_grid.max())
    ax.set_ylim(y_lim_low, y_lim_high)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("Energy")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.15)
    return minima


def draw_histogram(ax, theta_star, z, x_range, title):
    bins = np.linspace(x_range[0], x_range[1], 48)
    for g in ("N", "W", "S"):
        sub = theta_star[z == g]
        ax.hist(sub, bins=bins, color=GROUP_COLOR[g], alpha=0.7,
                edgecolor="white", lw=0.4,
                label=f"{g} (n={len(sub)})")
    for g_label, mu_val, color in [("S", PARAMS_BASE["mu_S"], C_S),
                                    ("W", PARAMS_BASE["mu_W"], C_W),
                                    ("N", PARAMS_BASE["mu_N"], C_N)]:
        ax.axvline(mu_val, color=color, ls="--", lw=1.3, alpha=0.85)
    ax.set_xlim(*x_range)
    ax.set_xlabel(r"$\theta^*$")
    ax.set_ylabel("count")
    ax.set_title(title, fontweight="bold", fontsize=10.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.15, axis="y")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    x_range = (-3.5, 4.0)

    for col, n in enumerate([200, 1500]):
        sigma_theta = SCALE_SIGMA[n]
        npz_path = os.path.join(_THIS_DIR, f"data_n{n}_seed0.npz")
        d = np.load(npz_path)
        theta_star = d["theta_star"]
        z = np.array([str(x) for x in d["z"]])
        conflict = int(d["conflict_mask"].sum())

        draw_landscape(
            axes[0, col], sigma_theta,
            f"(A{col + 1}) Energy landscape  n={n},  σ_θ={sigma_theta}\n"
            rf"μ=(+2.5, +1.0, −1.5), α=0.20, ρ=(0.2, 0.2, 0.6), "
            f"damping=0.85, conflict={conflict}/{n}"
        )
        draw_histogram(
            axes[1, col], theta_star, z, x_range,
            f"(B{col + 1}) θ* truth histogram  n={n}  "
            f"(S/W/N = {int((z=='S').sum())}/{int((z=='W').sum())}/{int((z=='N').sum())})"
        )

    fig.suptitle(
        "Study 1C — data distribution + target landscape  (n=200 vs n=1500)",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(_THIS_DIR, "data_landscape_overview.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
