"""
n 이 커질수록 AWSGLD 와 다른 샘플러의 격차가 벌어지는가?

각 패널 = 하나의 지표, x축 = n (200,1500,10000, log),
선 하나 = "AWSGLD advantage over (other)".
부호 통일: 위로 갈수록 항상 AWSGLD 가 더 좋음.
  - higher_is_better 지표: adv = AWSGLD - other
  - lower_is_better  지표: adv = other - AWSGLD
선이 우상향 = n 이 커질수록 격차 벌어짐.
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

NS = [200, 1500, 10000]
BASE = "AWSGLD"
OTHERS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC"]
COLORS = {"SGLD": "#888888", "qSGLD": "#1f77b4",
          "cycSGLD": "#2ca02c", "SGHMC": "#9467bd"}

# (key, 제목, higher_is_better)
PANELS = [
    ("ndcg", "nDCG@k", True),
    ("spearman", "Spearman ρ", True),
    ("mse_all", "MSE (all)", False),
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
    data = {n: json.load(open(f"results_n{n}_multiseed_summary.json"))["aggregated"]
            for n in NS}
    x = np.array(NS, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (key, title, hib) in zip(axes, PANELS):
        for model in OTHERS:
            adv, err = [], []
            for n in NS:
                aw_mu, aw_sd = get(data[n], BASE, key)
                mu, sd = get(data[n], model, key)
                d = (aw_mu - mu) if hib else (mu - aw_mu)
                adv.append(d)
                err.append(np.hypot(aw_sd, sd))
            ax.errorbar(x, adv, yerr=err, marker="o", capsize=4,
                        lw=2, ms=7, color=COLORS[model],
                        label=f"vs {model}", alpha=0.9)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel("n (데이터 크기)")
        ax.set_ylabel("AWSGLD 우위 (위=AWSGLD 우세)")
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="best", fontsize=10, framealpha=0.9)
    fig.suptitle(
        "n 이 커질수록 격차가 벌어지는가? — AWSGLD 우위의 n-추세\n"
        "(선이 우상향하면 n 증가에 따라 AWSGLD 격차 확대)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = "aw_gap_vs_n.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
