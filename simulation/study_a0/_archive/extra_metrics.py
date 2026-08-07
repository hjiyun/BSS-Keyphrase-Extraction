"""
Study 1B — 추가 지표 (Precision@k, ESS, Confusion matrix).

입력
----
- data_seed0.npz       : theta_star, z
- ava_results.npz      : acMH/AWSGLD theta_store, theta_hat
- sgld_results.npz     : SGLD/qSGLD/cycSGLD theta_store, theta_hat
- ava_metric_summary.json, sgld_metric_summary.json  : wall_time

3 지표
------
(1) Precision@k / Recall@k / F1@k
    - Truth top-k = θ* 의 상위 k 노드 (PU 의 진짜 키프레이즈 = S+W=160)
    - Method top-k = θ̂ 의 상위 k 노드
    - P@k = TP/k,  R@k(vs truth_top_k) = TP/k = P@k 동일,
      R@k(vs relevant set) = TP/|S+W|=TP/160
    - F1@k = 2PR/(P+R)
    - k ∈ {20, 50, 80, 160} 으로 sweep.

(2) ESS (Effective Sample Size) + cost-per-ESS
    - 각 노드의 chain 0 post-burn autocorrelation 기반 ESS 추정.
    - 노드별 ESS 분포를 median, q10, q90 으로 요약.
    - cost_per_ESS = wall_time_chain0 / median_ESS  (단위 시간당 정보량 역지표).

(3) Correct basin rate / Confusion matrix
    - 각 노드의 θ̂_i 를 가장 가까운 μ_g 로 분류 → predicted basin.
    - truth z_i × predicted basin 의 3x3 confusion matrix.
    - 대각선 = correct basin rate.

출력
----
- extra_metrics_summary.json
- extra_metrics_precision_at_k.png
- extra_metrics_ess.png
- extra_metrics_confusion.png
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
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from local_trap_landscape import PARAMS  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# 색상
METHOD_COLOR = {
    "acMH": "#E1B12C", "SGLD": "#2F6DB2", "qSGLD": "#D85A30",
    "cycSGLD": "#4E9A51", "AWSGLD": "#9B59B6",
}
GROUP_COLOR = {"S": "#2F6DB2", "W": "#D85A30", "N": "#6B6B6B"}
METHOD_ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "AWSGLD"]
K_VALUES = (20, 50, 80, 160)


# ──────────────────────────────────────────────────────────────────────────
# (1) Precision/Recall/F1 @ k
# ──────────────────────────────────────────────────────────────────────────
def precision_recall_f1_at_k(theta_star, theta_hat, z, k):
    """
    Truth top-k : θ* 상위 k 노드.
    Method top-k: θ̂ 상위 k 노드.
    Relevant set (PU-style): S+W = 160 노드.

    Returns dict with p@k, r@k_truth (=p@k), r@k_relevant, f1.
    """
    truth_top_k = set(np.argsort(theta_star)[::-1][:k].tolist())
    method_top_k = set(np.argsort(theta_hat)[::-1][:k].tolist())
    tp = len(truth_top_k & method_top_k)
    p_at_k = tp / k
    relevant_set = set(np.where((z == "S") | (z == "W"))[0].tolist())
    relevant_in_topk = len(method_top_k & relevant_set)
    r_relevant = relevant_in_topk / max(len(relevant_set), 1)
    p_relevant = relevant_in_topk / k
    f1_relevant = (2 * p_relevant * r_relevant / (p_relevant + r_relevant)
                   if (p_relevant + r_relevant) > 0 else 0.0)
    return {
        "k": k,
        "p_at_k": float(p_at_k),
        "r_at_k_truth_topk": float(p_at_k),  # same since |truth_top_k|=k
        "p_relevant_at_k": float(p_relevant),
        "r_relevant_at_k": float(r_relevant),
        "f1_relevant_at_k": float(f1_relevant),
    }


# ──────────────────────────────────────────────────────────────────────────
# (2) ESS per node (Geyer initial positive sequence estimator)
# ──────────────────────────────────────────────────────────────────────────
def _autocorr_fft(x):
    """1D autocorrelation via FFT, normalized to ρ(0)=1."""
    x = x - x.mean()
    n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0] + 1e-30
    return acf


def ess_per_node(theta_post_chain, max_lag=None):
    """
    Geyer initial positive sequence: sum positive ρ_k until negative pair sum.
    theta_post_chain shape (L, n).
    """
    L, n = theta_post_chain.shape
    if max_lag is None:
        max_lag = min(L // 3, 1000)
    ess_arr = np.full(n, np.nan)
    for i in range(n):
        x = theta_post_chain[:, i]
        if x.std() < 1e-10:
            ess_arr[i] = float(L)
            continue
        acf = _autocorr_fft(x)[:max_lag]
        # initial positive sequence: pair (ρ_{2k} + ρ_{2k+1}) > 0
        rho_sum = 0.0
        for k in range(1, max_lag // 2):
            pair = acf[2 * k] + acf[2 * k + 1]
            if pair <= 0:
                break
            rho_sum += pair
        ess_arr[i] = L / (1.0 + 2.0 * rho_sum)
    ess_arr = np.clip(ess_arr, 1.0, L)
    return ess_arr


# ──────────────────────────────────────────────────────────────────────────
# (3) Confusion matrix
# ──────────────────────────────────────────────────────────────────────────
def basin_assign(theta_hat, mu_map):
    """가장 가까운 μ_g 의 group 으로 할당."""
    n = len(theta_hat)
    centers = np.array([mu_map["S"], mu_map["W"], mu_map["N"]])
    labels = np.array(["S", "W", "N"])
    pred = []
    for v in theta_hat:
        idx = int(np.argmin(np.abs(centers - v)))
        pred.append(labels[idx])
    return np.array(pred)


def confusion_matrix(z, pred, groups=("S", "W", "N")):
    M = np.zeros((len(groups), len(groups)), dtype=int)
    for i, g_true in enumerate(groups):
        m = z == g_true
        for j, g_pred in enumerate(groups):
            M[i, j] = int(np.sum((pred[m] == g_pred)))
    return M  # rows=truth, cols=pred


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────
def load_all():
    data = np.load(os.path.join(_THIS_DIR, "data_seed0.npz"))
    ava = np.load(os.path.join(_THIS_DIR, "ava_results.npz"))
    sgld = np.load(os.path.join(_THIS_DIR, "sgld_results.npz"))
    with open(os.path.join(_THIS_DIR, "ava_metric_summary.json")) as f:
        ava_json = json.load(f)
    with open(os.path.join(_THIS_DIR, "sgld_metric_summary.json")) as f:
        sgld_json = json.load(f)

    theta_star = data["theta_star"]
    z = np.array([str(x) for x in data["z"]])

    theta_hat_by_m = {}
    chain0_post_by_m = {}
    wall_time_chain0 = {}
    BURN_IN = int(ava["BURN_IN"])

    for method in ("acMH", "AWSGLD"):
        theta_hat_by_m[method] = ava[f"{method}_theta_hat"]
        ts = ava[f"{method}_chain0_theta_store"]
        chain0_post_by_m[method] = ts[BURN_IN:]
        wall_time_chain0[method] = float(ava_json["wall_time_sec"][method][0])
    for method in ("SGLD", "qSGLD", "cycSGLD"):
        theta_hat_by_m[method] = sgld[f"{method}_theta_hat"]
        ts = sgld[f"{method}_chain0_theta_store"]
        chain0_post_by_m[method] = ts[BURN_IN:]
        wall_time_chain0[method] = float(sgld_json["wall_time_sec"][method][0])

    return theta_star, z, theta_hat_by_m, chain0_post_by_m, wall_time_chain0


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────
def plot_precision_at_k(pr_by_method, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("p_at_k", "Precision@k  (truth top-k 와 겹침)"),
        ("r_relevant_at_k", "Recall@k  (전체 키프레이즈 160개 대비)"),
        ("f1_relevant_at_k", "F1@k  (Recall vs relevant set 기반)"),
    ]
    for ax, (key, title) in zip(axes, metrics):
        for method in METHOD_ORDER:
            vals = [pr_by_method[method][k][key] for k in K_VALUES]
            ax.plot(K_VALUES, vals, marker="o", color=METHOD_COLOR[method],
                    label=method, lw=1.8, markersize=7)
        ax.set_xlabel("k"); ax.set_ylabel(key)
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0, 1.02)
        ax.grid(True, alpha=0.2); ax.legend(fontsize=9)
    fig.suptitle("Precision/Recall/F1 @ k  (top-k 키프레이즈 추출 task)",
                 fontweight="bold", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_ess(ess_summary, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    methods = METHOD_ORDER
    medians = [ess_summary[m]["ess_median"] for m in methods]
    q10s = [ess_summary[m]["ess_q10"] for m in methods]
    q90s = [ess_summary[m]["ess_q90"] for m in methods]
    x = np.arange(len(methods))
    ax.bar(x, medians, color=[METHOD_COLOR[m] for m in methods], alpha=0.85)
    ax.errorbar(x, medians,
                yerr=[np.array(medians) - np.array(q10s),
                      np.array(q90s) - np.array(medians)],
                fmt="none", color="black", capsize=4, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("ESS (per node)  median ± [q10, q90]")
    ax.set_title("Effective Sample Size  (T_post=4500)",
                 fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.2, axis="y")

    ax = axes[1]
    cost = [ess_summary[m]["cost_per_ess_sec"] for m in methods]
    ax.bar(x, cost, color=[METHOD_COLOR[m] for m in methods], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15)
    ax.set_ylabel("wall_time / median_ESS  [sec per ESS]")
    ax.set_yscale("log")
    ax.set_title("Cost per ESS  (낮을수록 단위 시간 당 정보량 ↑)",
                 fontweight="bold", fontsize=10.5)
    ax.grid(True, alpha=0.2, axis="y")
    for i, v in enumerate(cost):
        ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Chain 효율 — Effective Sample Size & Cost-per-ESS",
                 fontweight="bold", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_confusion(cm_by_method, out_path):
    n_m = len(METHOD_ORDER)
    fig, axes = plt.subplots(1, n_m, figsize=(3.4 * n_m, 4.5),
                             sharey=True)
    groups = ("S", "W", "N")
    for ax, method in zip(axes, METHOD_ORDER):
        cm = cm_by_method[method]
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / np.maximum(row_sum, 1)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(groups)):
            for j in range(len(groups)):
                txt = f"{cm[i, j]}\n({cm_norm[i, j]:.2f})"
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=9, color=color)
        ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups)
        ax.set_yticks(range(len(groups))); ax.set_yticklabels(groups)
        ax.set_xlabel("predicted basin")
        ax.set_ylabel("truth z" if method == METHOD_ORDER[0] else "")
        acc = np.trace(cm) / max(cm.sum(), 1)
        ax.set_title(f"{method}  (basin acc = {acc:.2%})",
                     fontweight="bold", fontsize=10.5,
                     color=METHOD_COLOR[method])
    fig.suptitle(
        "Confusion matrix — truth z × predicted basin (nearest μ_g)\n"
        "대각선 = 자기 basin 으로 옳게 안착한 노드 수 / 비율",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    theta_star, z, theta_hat_by_m, chain0_post_by_m, wall_time = load_all()
    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}

    print(f"n={len(theta_star)},  S/W/N = "
          f"{int((z=='S').sum())}/{int((z=='W').sum())}/{int((z=='N').sum())}")
    print()

    # ── (1) Precision/Recall/F1 @ k ──
    print("=" * 78)
    print("(1) Precision/Recall/F1 @ k")
    print("=" * 78)
    pr_by_method = {}
    for method in METHOD_ORDER:
        pr_by_method[method] = {}
        line = f"[{method:<7}] "
        for k in K_VALUES:
            pr = precision_recall_f1_at_k(theta_star, theta_hat_by_m[method], z, k)
            pr_by_method[method][k] = pr
            line += (f"k={k}: P={pr['p_at_k']:.3f} "
                     f"R_rel={pr['r_relevant_at_k']:.3f} "
                     f"F1={pr['f1_relevant_at_k']:.3f} | ")
        print(line)
    print()

    # ── (2) ESS ──
    print("=" * 78)
    print("(2) ESS (per node, chain 0 post-burn)")
    print("=" * 78)
    ess_summary = {}
    for method in METHOD_ORDER:
        ess = ess_per_node(chain0_post_by_m[method])
        med = float(np.median(ess))
        q10 = float(np.quantile(ess, 0.10))
        q90 = float(np.quantile(ess, 0.90))
        cost = wall_time[method] / max(med, 1e-9)
        ess_summary[method] = {
            "ess_median": med, "ess_q10": q10, "ess_q90": q90,
            "wall_time_chain0": wall_time[method],
            "cost_per_ess_sec": cost,
        }
        print(f"[{method:<7}] ESS median={med:>7.1f}  "
              f"q10={q10:>6.1f}  q90={q90:>7.1f}  "
              f"wall={wall_time[method]:>6.1f}s  "
              f"cost_per_ESS={cost:.4f}s")
    print()

    # ── (3) Confusion ──
    print("=" * 78)
    print("(3) Confusion matrix (truth z × predicted basin)")
    print("=" * 78)
    cm_by_method = {}
    for method in METHOD_ORDER:
        pred = basin_assign(theta_hat_by_m[method], mu_map)
        cm = confusion_matrix(z, pred)
        cm_by_method[method] = cm
        acc = np.trace(cm) / max(cm.sum(), 1)
        print(f"[{method}]  basin accuracy = {acc:.2%}")
        print(f"           pred:  S    W    N")
        for i, g in enumerate(("S", "W", "N")):
            print(f"  truth {g}:    {cm[i, 0]:>3}  {cm[i, 1]:>3}  {cm[i, 2]:>3}")
        print()

    # ── plots & save ──
    plot_precision_at_k(pr_by_method,
                        os.path.join(_THIS_DIR, "extra_metrics_precision_at_k.png"))
    plot_ess(ess_summary, os.path.join(_THIS_DIR, "extra_metrics_ess.png"))
    plot_confusion(cm_by_method,
                   os.path.join(_THIS_DIR, "extra_metrics_confusion.png"))

    summary = {
        "k_values": list(K_VALUES),
        "precision_recall_f1": pr_by_method,
        "ess": ess_summary,
        "confusion_matrix": {
            m: {"matrix": cm_by_method[m].tolist(),
                "basin_accuracy": float(np.trace(cm_by_method[m]) / max(cm_by_method[m].sum(), 1)),
                "rows_truth": ["S", "W", "N"], "cols_pred": ["S", "W", "N"]}
            for m in METHOD_ORDER
        },
    }
    out_json = os.path.join(_THIS_DIR, "extra_metrics_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {out_json}")


if __name__ == "__main__":
    main()
