"""
Study 1B — 데이터 분포 + local trap 한 눈에 보기.

2 panel 구성
-------------
(A) target 1D energy landscape (-log mixture posterior).
    - local_trap_landscape.py 와 동일 곡선.
    - y_lim 은 boundary blow-up 무시하고 inter-mode peak 기준으로 자름
      (barrier 높이가 한 눈에 보이게).
    - 인접 basin 간 ΔE 를 화살표로 명시.

(B) 실제 θ* truth histogram by group.
    - data_seed{N}.npz 의 theta_star 가 (A) 의 3 basin 위치와 정렬되는지 확인.

출력
----
- data_landscape_overview.png

실행
----
python3 simulation/study_1b/data_landscape_overview.py
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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from local_trap_landscape import PARAMS, E_mix  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


C_S, C_W, C_N = "#2F6DB2", "#D85A30", "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}


def main(seed=0):
    data_path = os.path.join(_THIS_DIR, f"data_seed{seed}.npz")

    d = np.load(data_path)
    theta_star = d["theta_star"]
    z = np.array([str(x) for x in d["z"]])
    n_total = int(d["n_total"])
    conflict = int(d["conflict_mask"].sum())
    damping = float(d["damping"])
    p_in, p_out = float(d["p_in"]), float(d["p_out"])

    mu_S, mu_W, mu_N = PARAMS["mu_S"], PARAMS["mu_W"], PARAMS["mu_N"]

    # ── target landscape 곡선 ──
    theta_grid = np.linspace(-3.5, 4.0, 1200)
    sigma2 = PARAMS["sigma_theta"] ** 2
    E_grid = E_mix(theta_grid, PARAMS["alpha"],
                   PARAMS["mu_S"], PARAMS["mu_W"], PARAMS["mu_N"],
                   PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"],
                   sigma2)

    # local minima / maxima 검출 (inter-mode peak 만 — boundary blow-up 무시)
    min_idx = argrelmin(E_grid)[0]
    max_idx = argrelmax(E_grid)[0]
    minima = [(float(theta_grid[i]), float(E_grid[i])) for i in min_idx]
    maxima = [(float(theta_grid[i]), float(E_grid[i])) for i in max_idx]

    # y_lim 잘라내기: local 최고 peak + 여유분 (boundary blow-up 무시)
    peak_E = max(p[1] for p in maxima) if maxima else float(E_grid.max())
    y_lim_high = peak_E + 1.5
    y_lim_low = max((min(p[1] for p in minima) if minima else 0.0) - 0.5, 0.0)

    # ── plot ──
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.1], hspace=0.30)

    # (A) Energy landscape — y_lim 캡
    axA = fig.add_subplot(gs[0])
    E_show = np.where(E_grid > y_lim_high, np.nan, E_grid)
    axA.plot(theta_grid, E_show, color="#1D9E75", lw=2.6, zorder=3)
    for g_label, mu_val, color in [("S", mu_S, C_S),
                                    ("W", mu_W, C_W),
                                    ("N", mu_N, C_N)]:
        axA.axvline(mu_val, color=color, ls="--", lw=1.4, alpha=0.85)
        axA.axvspan(mu_val - 0.6, mu_val + 0.6, alpha=0.08, color=color)
        axA.text(mu_val, y_lim_high * 0.94, g_label, ha="center", color=color,
                 fontsize=13, fontweight="bold")

    # local minima 마커
    for x_m, y_m in minima:
        axA.plot(x_m, y_m, marker="v", color="#1D9E75", markersize=11,
                 markeredgecolor="black", markeredgewidth=0.5, zorder=5)
        axA.annotate(f"E={y_m:.2f}\nθ={x_m:+.2f}",
                     xy=(x_m, y_m), xytext=(x_m, y_m - 0.35),
                     fontsize=8, ha="center", va="top", color="#1D9E75")

    sorted_minima = sorted(minima, key=lambda m: m[0])

    axA.set_xlim(theta_grid.min(), theta_grid.max())
    axA.set_ylim(y_lim_low, y_lim_high)
    axA.set_xlabel(r"$\theta$")
    axA.set_ylabel(r"Energy  $-\log p(\theta\,|\,Y, \alpha)$")
    axA.set_title(
        f"(A) Target 분포의 1D mixture posterior energy  "
        f"[y-axis: boundary blow-up 제외, local peak 까지만 표시]\n"
        rf"$\mu$=({mu_S}, {mu_W}, {mu_N}),  $\sigma_\theta$={PARAMS['sigma_theta']},  "
        rf"$\alpha$={PARAMS['alpha']},  $\rho$=({PARAMS['rho_S']}, {PARAMS['rho_W']}, {PARAMS['rho_N']})  "
        "—  ▼ local min",
        fontweight="bold", fontsize=10.5, pad=8,
    )
    axA.grid(True, alpha=0.15)

    # (B) θ* truth histogram by group
    axB = fig.add_subplot(gs[1], sharex=axA)
    bins = np.linspace(theta_grid.min(), theta_grid.max(), 48)
    for g in ("N", "W", "S"):
        sub = theta_star[z == g]
        axB.hist(sub, bins=bins, color=GROUP_COLOR[g], alpha=0.65,
                 label=f"{g} (n={len(sub)})", edgecolor="white", lw=0.4)
    for g_label, mu_val, color in [("S", mu_S, C_S),
                                    ("W", mu_W, C_W),
                                    ("N", mu_N, C_N)]:
        axB.axvline(mu_val, color=color, ls="--", lw=1.4, alpha=0.85)
    axB.set_ylabel("count")
    axB.set_xlabel(r"$\theta$")
    axB.set_title(
        f"(B) data_seed{seed}.npz 의 실제 θ* truth — group 별 sampling 결과 "
        "(위 (A) 의 3 basin 위치와 정렬되어야 함)\n"
        f"n={n_total}, n_S=80 / n_W=80 / n_N=240,  σ_θ={PARAMS['sigma_theta']} 의 group-Normal sampling",
        fontweight="bold", fontsize=10.5, pad=6,
    )
    axB.legend(loc="upper right", fontsize=9)
    axB.grid(True, alpha=0.15, axis="y")

    fig.suptitle(
        f"Study 1B — 데이터 분포 + local trap landscape  (seed={seed})\n"
        f"graph: n={n_total}, p_in={p_in}, p_out={p_out}, damping={damping},  "
        f"label conflict: {conflict}/{n_total} 노드",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(_THIS_DIR, "data_landscape_overview.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out}")

    # 텍스트로도 barrier 출력
    print()
    print("Local minima:")
    for x_m, y_m in minima:
        print(f"  θ={x_m:+.3f}  E={y_m:.3f}")
    print("Inter-basin peaks (ΔE):")
    for x_p, y_p in maxima:
        left = [m for m in sorted_minima if m[0] < x_p]
        right = [m for m in sorted_minima if m[0] > x_p]
        if not left or not right:
            continue
        m_L = max(left, key=lambda m: m[0])
        m_R = min(right, key=lambda m: m[0])
        print(f"  peak θ={x_p:+.3f}, E={y_p:.3f}  "
              f"|  ΔE_left={(y_p - m_L[1]):.3f}  ΔE_right={(y_p - m_R[1]):.3f}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
