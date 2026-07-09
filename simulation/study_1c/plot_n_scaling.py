"""
n 스케일링 비교: n=200, 1500, 10000 에 따라 샘플러별 성능 지표가
어떻게 변하는지(특히 모델 간 격차가 커지는지) 시각화한다.

각 패널 = 하나의 지표, x축 = n (log scale), 선 하나 = 하나의 샘플러.
mean ± std (5 seeds) 를 errorbar 로 표시.
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
MODELS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]

# 모델별 색상 (AWSGLD 강조)
COLORS = {
    "SGLD": "#888888",
    "qSGLD": "#1f77b4",
    "cycSGLD": "#2ca02c",
    "SGHMC": "#9467bd",
    "AWSGLD": "#d62728",
}

# 그릴 지표와 표시 정보: (key, 제목, log_y 여부)
# ndcg 는 n마다 키 이름이 달라 특별 처리("ndcg")
PANELS = [
    ("ndcg", "nDCG@k  (높을수록 좋음)", False),
    ("spearman", "Spearman ρ  (높을수록 좋음)", False),
    ("mse_all", "MSE (all)  (낮을수록 좋음)", False),
    ("ess_median", "ESS median  (높을수록 좋음)", True),
    ("cost_per_ess", "cost per ESS  (낮을수록 좋음)", True),
    ("R_hat_max", "R-hat max  (1에 가까울수록 좋음)", False),
]


def load_all():
    data = {}
    for n in NS:
        with open(f"results_n{n}_multiseed_summary.json") as f:
            data[n] = json.load(f)["aggregated"]
    return data


def get_metric(agg, model, key):
    """key=='ndcg' 면 ndcg_at_* 를 찾아 반환. (mean, std)"""
    m = agg[model]
    if key == "ndcg":
        key = next(k for k in m if k.startswith("ndcg_at_"))
    return m[key]["mean"], m[key]["std"]


def main():
    data = load_all()
    x = np.array(NS, dtype=float)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    for ax, (key, title, logy) in zip(axes, PANELS):
        for model in MODELS:
            means, stds = [], []
            for n in NS:
                mu, sd = get_metric(data[n], model, key)
                means.append(mu)
                stds.append(sd)
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(
                x, means, yerr=stds,
                marker="o", capsize=4, lw=2, ms=7,
                color=COLORS[model], label=model,
                alpha=0.9,
            )
        ax.set_xscale("log")
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("n (데이터 크기)")
        ax.set_title(title, fontsize=12)
        ax.grid(True, which="both", alpha=0.3)

    axes[0].legend(loc="best", fontsize=10, framealpha=0.9)
    fig.suptitle(
        "샘플러 성능의 n-스케일링 (n=200 → 1500 → 10000, mean±std over 5 seeds)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = "n_scaling_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
