"""
n=200 에서 AWSGLD 기준으로 다른 샘플러와의 차이를 본다.
각 지표마다  (AWSGLD_mean - other_mean)  을 막대로 표시.

막대 색:
  - 초록 = AWSGLD 가 더 좋음 (방향 고려)
  - 빨강 = AWSGLD 가 더 나쁨
errorbar = 두 표본 std 의 결합(sqrt(s_aw^2 + s_other^2)).
"""
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager

_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(_FONT)
matplotlib.rcParams["font.family"] = font_manager.FontProperties(fname=_FONT).get_name()
matplotlib.rcParams["axes.unicode_minus"] = False

N = 200
BASE = "AWSGLD"
OTHERS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC"]

# (key, 제목, higher_is_better)
PANELS = [
    ("ndcg", "nDCG@40", True),
    ("spearman", "Spearman ρ", True),
    ("mse_all", "MSE (all)", False),
    ("mse_S", "MSE (S)", False),
    ("mse_W", "MSE (W)", False),
    ("mse_N", "MSE (N)", False),
    ("ess_median", "ESS median", True),
    ("cost_per_ess", "cost per ESS", False),
    ("R_hat_max", "R-hat max", False),
]


def get(agg, model, key):
    m = agg[model]
    if key == "ndcg":
        key = next(k for k in m if k.startswith("ndcg_at_"))
    return m[key]["mean"], m[key]["std"]


def main():
    agg = json.load(open(f"results_n{N}_multiseed_summary.json"))["aggregated"]

    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.ravel()
    y = np.arange(len(OTHERS))

    for ax, (key, title, hib) in zip(axes, PANELS):
        aw_mu, aw_sd = get(agg, BASE, key)
        diffs, errs, colors = [], [], []
        for model in OTHERS:
            mu, sd = get(agg, model, key)
            d = aw_mu - mu  # AWSGLD - other
            diffs.append(d)
            errs.append(np.hypot(aw_sd, sd))
            # AWSGLD 가 더 좋은가? higher_is_better 이면 d>0 이 좋음
            aw_better = (d > 0) if hib else (d < 0)
            colors.append("#2ca02c" if aw_better else "#d62728")

        ax.barh(y, diffs, xerr=errs, color=colors, alpha=0.85,
                capsize=4, edgecolor="black", lw=0.5)
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(y)
        ax.set_yticklabels(OTHERS)
        ax.invert_yaxis()
        arrow = "↑좋음" if hib else "↓좋음"
        ax.set_title(f"{title}   ({arrow})", fontsize=12)
        ax.set_xlabel(f"AWSGLD − other")
        ax.grid(True, axis="x", alpha=0.3)
        # 막대 끝에 값 표기
        for yi, d in zip(y, diffs):
            ax.text(d, yi, f" {d:+.3g}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9)

    fig.suptitle(
        f"n={N}: AWSGLD 기준 다른 샘플러와의 지표 차이 (AWSGLD − other, mean over 5 seeds)\n"
        "초록 = AWSGLD 우세, 빨강 = AWSGLD 열세",
        fontsize=14, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = f"aw_diff_n{N}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
