"""
Study 1B — paper-ready figures.

(A) θ̂ histogram (truth vs acMH vs AWSGLD)
    3-mode 분리 회복 여부를 한 장에.

(B) π̂ keyphrase ranking
    truth 의 π* 순으로 정렬한 phrase index 에서 각 method 의 π̂ 곡선.

입력
----
- data_seed0.npz   (theta_star, Y, z, conflict_mask)
- ava_results.npz  (acMH_theta_hat, AWSGLD_theta_hat)

출력
----
- fig_theta_histograms.png
- fig_pi_ranking.png

실행
----
python3 simulation/study_1b/fig_recovery_paper.py
"""
import os
import sys

import numpy as np
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from local_trap_landscape import PARAMS, sigmoid  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


C_S, C_W, C_N = "#2F6DB2", "#D85A30", "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}
METHOD_COLOR = {"truth": "#222222", "acMH": "#E1B12C", "AWSGLD": "#9B59B6"}

GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)


def alpha_lk(theta, Y, alpha):
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    return float(np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp)))


def alpha_find(theta, Y, grid):
    return float(grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))])


def figure_A_theta_histograms(theta_star, theta_hats, z, mu_map, out_path,
                              density=False):
    """3-panel stacked: truth / acMH / AWSGLD.

    density=False → 'count'  (default)
    density=True  → 'density' (각 group 마다 면적=1 정규화 → group 크기 차이 무시)
    """
    fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)

    x_lo = min(theta_star.min(), *[t.min() for t in theta_hats.values()]) - 0.4
    x_hi = max(theta_star.max(), *[t.max() for t in theta_hats.values()]) + 0.4
    bins = np.linspace(x_lo, x_hi, 60)

    panels = [
        ("truth  $\\theta^*$  (3-mode mixture 에서 sampling)", theta_star, "truth"),
        ("acMH  $\\hat\\theta$  (bad init 에서 chain mean)", theta_hats["acMH"], "acMH"),
        ("AWSGLD  $\\hat\\theta$  (bad init 에서 chain mean)", theta_hats["AWSGLD"], "AWSGLD"),
    ]
    for ax, (title, theta_vec, method) in zip(axes, panels):
        for g in ("N", "W", "S"):
            m = z == g
            ax.hist(theta_vec[m], bins=bins, color=GROUP_COLOR[g], alpha=0.7,
                    edgecolor="white", lw=0.4, density=density,
                    label=f"{g} (n={int(m.sum())})")
        for g, mu_val in mu_map.items():
            ax.axvline(mu_val, color=GROUP_COLOR[g], ls="--", lw=1.4, alpha=0.85)

        # group means as crosses (실제 method 가 어디 안착했는지 표시)
        ymax = ax.get_ylim()[1]
        for g in ("S", "W", "N"):
            m = z == g
            gm = float(theta_vec[m].mean())
            ax.plot(gm, ymax * 0.88, marker="x", color=GROUP_COLOR[g],
                    markersize=10, markeredgewidth=2.4, zorder=6)
            ax.text(gm, ymax * 0.95, f"{gm:+.2f}", color=GROUP_COLOR[g],
                    fontsize=8, ha="center", fontweight="bold")

        ax.set_title(title, fontweight="bold", fontsize=11, pad=6)
        ax.set_ylabel("density" if density else "count")
        ax.grid(True, alpha=0.15, axis="y")
        if method == "truth":
            ax.legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel(r"$\theta$")
    suptitle_mode = ("density (group 별 면적=1 정규화)" if density else "count")
    fig.suptitle(
        f"Figure A — $\\theta$ histogram [{suptitle_mode}]: "
        "truth 의 3-mode vs sampler 의 회복도\n"
        "▽ 점선=각 group $\\mu_g$,  ×=method 의 group 별 mean($\\hat\\theta_g$)",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def figure_C_rank_qq(theta_star, theta_hats, z, out_path, spear_by_method):
    """Q-Q plot 스타일: truth rank vs sampler rank."""
    rank_truth = np.argsort(np.argsort(theta_star)) + 1  # 1..n
    n = len(theta_star)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for ax, method in zip(axes, ("acMH", "AWSGLD")):
        rank_m = np.argsort(np.argsort(theta_hats[method])) + 1
        for g in ("N", "W", "S"):
            mask = z == g
            ax.scatter(rank_truth[mask], rank_m[mask],
                       color=GROUP_COLOR[g], s=22, alpha=0.7,
                       label=f"{g} (n={int(mask.sum())})", zorder=3)
        ax.plot([1, n], [1, n], color="black", ls="--", lw=1.2, alpha=0.7,
                zorder=2, label="perfect (y=x)")
        ax.set_xlim(0, n + 1); ax.set_ylim(0, n + 1)
        ax.set_xlabel(r"rank of truth $\theta^*$")
        ax.set_ylabel(rf"rank of {method} $\hat\theta$")
        sp = spear_by_method[method]
        ax.set_title(
            f"{method}  (Spearman = {sp:.3f})",
            fontweight="bold", fontsize=11,
        )
        ax.grid(True, alpha=0.18)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.92)
        ax.set_aspect("equal")

    fig.suptitle(
        "Figure C — rank Q-Q plot:  truth rank vs sampler rank\n"
        "대각선에 붙으면 ranking 회복 ↑, 무작위로 흩어지면 회복 ×",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def figure_D_pi_group_boxplot(pi_star, pi_hats, z, out_path):
    """3 panel stacked: truth / acMH / AWSGLD 각각 S/W/N boxplot."""
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9), sharex=True, sharey=True)
    panels = [
        ("truth  $\\pi^*$  (α_true=0.20)", pi_star, "truth"),
        ("acMH  $\\hat\\pi$  ($\\hat{\\alpha}$=0.500)", pi_hats["acMH"], "acMH"),
        ("AWSGLD  $\\hat\\pi$  ($\\hat{\\alpha}$=0.500)", pi_hats["AWSGLD"], "AWSGLD"),
    ]
    groups = ("S", "W", "N")
    positions = [1, 2, 3]
    for ax, (title, pi_vec, method) in zip(axes, panels):
        data_by_g = [pi_vec[z == g] for g in groups]
        bp = ax.boxplot(
            data_by_g, positions=positions, widths=0.55,
            patch_artist=True, showfliers=True,
            medianprops=dict(color="black", lw=1.4),
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(GROUP_COLOR[g])
            patch.set_alpha(0.65)
        # group means as text
        for x_pos, g in zip(positions, groups):
            gm = float(pi_vec[z == g].mean())
            ax.text(x_pos, gm, f"mean={gm:.3f}", color=GROUP_COLOR[g],
                    fontsize=8, ha="center", va="bottom", fontweight="bold")
        ax.set_xticks(positions)
        ax.set_xticklabels([f"{g} (n={int((z == g).sum())})" for g in groups])
        ax.set_ylabel(r"$\pi$")
        ax.set_title(title, fontweight="bold", fontsize=11, pad=6)
        ax.grid(True, alpha=0.18, axis="y")

    fig.suptitle(
        "Figure D — π distribution by group (S / W / N)\n"
        "truth: 세 box 명확히 분리.  method 가 분리 회복했는지 시각 확인.",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def figure_B_pi_ranking(pi_star, pi_hats, theta_star, out_path,
                        spear_by_method):
    """truth 의 π* 순으로 정렬한 phrase index 에서 π̂ 곡선."""
    order = np.argsort(pi_star)[::-1]
    x = np.arange(len(pi_star))

    fig, ax = plt.subplots(figsize=(11, 5.8))

    ax.plot(x, pi_star[order], color=METHOD_COLOR["truth"], lw=2.4, zorder=4,
            label=fr"truth  $\pi^*$  (α=0.20, n={len(pi_star)})")
    for method in ("acMH", "AWSGLD"):
        sp = spear_by_method[method]
        ax.plot(x, pi_hats[method][order], color=METHOD_COLOR[method],
                lw=1.5, alpha=0.92,
                label=fr"{method}  $\hat\pi$   (Spearman $\hat\theta$↔$\theta^*$ = {sp:.3f})")
    ax.set_xlabel("phrase index  (truth $\\pi^*$ 의 내림차순으로 정렬)")
    ax.set_ylabel(r"$\pi = (1-\alpha) \cdot \sigma(\theta)$")
    ax.set_title(
        "Figure B — keyphrase ranking 회복도  \n"
        "x축: truth $\\pi^*$ 순.  truth = 계단형 (S→W→N).  "
        "method 가 ranking 을 살려두면 같은 계단을 따라감, 무너지면 flat.",
        fontweight="bold", fontsize=11, pad=8,
    )
    ax.set_ylim(0, max(pi_star.max(), *[p.max() for p in pi_hats.values()]) * 1.08)
    ax.set_xlim(0, len(pi_star) - 1)
    ax.grid(True, alpha=0.18)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def main():
    data = np.load(os.path.join(_THIS_DIR, "data_seed0.npz"))
    ava = np.load(os.path.join(_THIS_DIR, "ava_results.npz"))

    theta_star = data["theta_star"]
    Y = data["Y"]
    z = np.array([str(x) for x in data["z"]])

    theta_hats = {
        "acMH": ava["acMH_theta_hat"],
        "AWSGLD": ava["AWSGLD_theta_hat"],
    }

    # truth π* 는 데이터 생성 시 사용한 α=0.20 으로
    alpha_true = float(data["param_alpha"])
    pi_star = (1.0 - alpha_true) * sigmoid(theta_star)

    # method 별 π̂ : alpha 는 alpha_find(theta_hat, Y, GRID) 로 추정 (sampler 와 동일 로직)
    pi_hats = {}
    alpha_hats = {}
    spear_by_method = {}
    for method, th in theta_hats.items():
        a = alpha_find(th, Y, GRID)
        alpha_hats[method] = a
        pi_hats[method] = (1.0 - a) * sigmoid(th)
        spear_by_method[method] = float(spearmanr(theta_star, th).statistic)

    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}

    print(f"alpha_true = {alpha_true:.3f}")
    for m, a in alpha_hats.items():
        print(f"alpha_hat({m}) = {a:.3f}")
    for m, sp in spear_by_method.items():
        print(f"Spearman({m}) = {sp:.3f}")

    figure_A_theta_histograms(
        theta_star, theta_hats, z, mu_map,
        out_path=os.path.join(_THIS_DIR, "fig_theta_histograms.png"),
        density=False,
    )
    figure_A_theta_histograms(
        theta_star, theta_hats, z, mu_map,
        out_path=os.path.join(_THIS_DIR, "fig_theta_histograms_density.png"),
        density=True,
    )
    figure_B_pi_ranking(
        pi_star, pi_hats, theta_star,
        out_path=os.path.join(_THIS_DIR, "fig_pi_ranking.png"),
        spear_by_method=spear_by_method,
    )
    figure_C_rank_qq(
        theta_star, theta_hats, z,
        out_path=os.path.join(_THIS_DIR, "fig_rank_qq.png"),
        spear_by_method=spear_by_method,
    )
    figure_D_pi_group_boxplot(
        pi_star, pi_hats, z,
        out_path=os.path.join(_THIS_DIR, "fig_pi_group_boxplot.png"),
    )


if __name__ == "__main__":
    main()
