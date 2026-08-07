"""
Study 1B — NDCG@k 계산 (graded ranking quality).

정의 (study_1a/langevin_methods_comparison.py 의 ndcg_at_k 와 동일)
--------------------------------------------------------------
- rel_i = rank(theta_star_i) / (n-1)   ∈ [0, 1]  (rank-based graded relevance)
- DCG@k  = Σ_{i=1..k} (2^rel_i - 1) / log2(i+1)   (method 의 top-k 순서 기준)
- IDCG@k = 동일 식, ideal (truth) top-k 순서 기준
- NDCG@k = DCG@k / IDCG@k    ∈ [0, 1]

NDCG@k 가 F1@k 보다 우월한 점
- top-k 안에서 "어느 순서로 배치했는가" 까지 점수에 반영 (graded)
- F1@k 는 binary (in / out) 만 봄

입력
- data_seed0.npz, ava_results.npz, sgld_results.npz

출력
- ndcg_summary.json,  ndcg_at_k.png
"""
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()


METHOD_COLOR = {
    "acMH": "#E1B12C", "SGLD": "#2F6DB2", "qSGLD": "#D85A30",
    "cycSGLD": "#4E9A51", "AWSGLD": "#9B59B6",
}
METHOD_ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "AWSGLD"]
K_VALUES = (10, 20, 50, 80, 160)


def ndcg_at_k(theta_star, theta_hat, k):
    rel = np.argsort(np.argsort(theta_star)).astype(float)
    rel = rel / max(len(theta_star) - 1, 1)
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    ideal_rank = np.argsort(theta_star)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[pred_rank] - 1) * discounts)
    idcg = np.sum((2 ** rel[ideal_rank] - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def main():
    parent = os.path.dirname(_THIS_DIR)  # study_1b
    data = np.load(os.path.join(parent, "data_seed0.npz"))
    ava = np.load(os.path.join(parent, "ava_results.npz"))
    sgld = np.load(os.path.join(parent, "sgld_results.npz"))
    theta_star = data["theta_star"]

    theta_hat_by_m = {}
    for method in ("acMH", "AWSGLD"):
        theta_hat_by_m[method] = ava[f"{method}_theta_hat"]
    for method in ("SGLD", "qSGLD", "cycSGLD"):
        theta_hat_by_m[method] = sgld[f"{method}_theta_hat"]

    print(f"NDCG@k for k ∈ {K_VALUES}")
    print()
    header = f"{'Method':<8}  " + "".join([f"NDCG@{k:<3} " for k in K_VALUES])
    print(header)
    print("-" * len(header))
    results = {}
    for method in METHOD_ORDER:
        row = {}
        for k in K_VALUES:
            v = ndcg_at_k(theta_star, theta_hat_by_m[method], k)
            row[k] = v
        results[method] = row
        line = f"{method:<8}  " + "".join([f"{row[k]:>7.4f}  " for k in K_VALUES])
        print(line)

    # plot
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for method in METHOD_ORDER:
        vals = [results[method][k] for k in K_VALUES]
        ax.plot(K_VALUES, vals, marker="o", lw=1.8, markersize=7,
                color=METHOD_COLOR[method], label=method)
    ax.set_xlabel("k"); ax.set_ylabel("NDCG@k")
    ax.set_xticks(K_VALUES)
    ax.set_ylim(0, 1.02)
    ax.set_title("NDCG@k  (graded ranking quality, rank-based relevance)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.2); ax.legend(fontsize=10)
    fig.tight_layout()
    out_png = os.path.join(parent, "ndcg_at_k.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved -> {out_png}")

    out_json = os.path.join(parent, "ndcg_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"k_values": list(K_VALUES), "ndcg_by_method": results},
                  f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
