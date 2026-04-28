"""
Controlled simulation for comparing AWSGLD vs acMH on theta / pi recovery.

Design goals
------------
1. Synthetic graph and synthetic truth are fully controlled.
2. True theta* and pi* are known.
3. Initialization follows the original BSS paper / R code:
   - u_0 = solve(B) %*% rep(1-d, n)
   - Base_Line = solve(B_star) %*% Y
   - ini = Base_to_start(Base_Line)
   - alpha_est = alpha_find(u_0, Y, grid)
4. acMH uses the existing Python implementation in keyphrase_functions.py.
5. AWSGLD is left as a plug-in hook because its sampler is not implemented yet.

Run
---
python3 /home/jiyoon/3차/BSS-Keyphrase-Extraction-master/simlation/awsgld_acmh_theta_recovery_simulation.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy.stats import spearmanr, kendalltau


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
ORIG_DIR = os.path.join(CODE_DIR, "original")
for _p in (CODE_DIR, ORIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from keyphrase_functions import (  # noqa: E402
    inv_logit,
    base_to_start,
    alpha_find,
    gibbs_mh,
)
from keyphrase_functions_awsgld_0422 import gibbs_mh as gibbs_mh_awsgld  # noqa: E402


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DAMPING = 0.85
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

T = 5000
BURN_IN = 1000
R = 3
SEED_BASE = 20260422

BLOCK_PROBS = {
    "within": 0.20,
    "between_sw": 0.03,
    "between_other": 0.005,
}

DEFAULT_SCENARIO = {
    "name": "ControlledSparse_v2",
    "n_total": 400,
    "rho_S": 0.10,
    "rho_W": 0.18,
    "rho_N": 0.72,
    "mu_S": 2.0,
    "mu_W": 1.0,
    "mu_N": -1.0,
    "sigma_theta": 0.55,
    "alpha_true": 0.40,
}


@dataclass
class TrialResult:
    method: str
    mse_theta: float
    mse_pi: float
    mse_calibrated: float
    spearman: float
    kendall: float
    topk_overlap: float
    ndcg_at_k: float
    alpha_hat: float
    n_obs: int
    wall_time_sec: float
    theta_hat: np.ndarray
    pi_hat: np.ndarray


def build_block_graph(scenario, rng):
    """
    Build a 3-block graph with S/W/N communities.
    Group membership is known and controlled.
    """
    n = scenario["n_total"]
    n_s = int(round(n * scenario["rho_S"]))
    n_w = int(round(n * scenario["rho_W"]))
    n_n = n - n_s - n_w

    group = np.array(["S"] * n_s + ["W"] * n_w + ["N"] * n_n, dtype="<U1")
    perm = rng.permutation(n)
    group = group[perm]

    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            gi, gj = group[i], group[j]
            if gi == gj:
                p = BLOCK_PROBS["within"]
            elif {gi, gj} == {"S", "W"}:
                p = BLOCK_PROBS["between_sw"]
            else:
                p = BLOCK_PROBS["between_other"]
            if rng.uniform() < p:
                A[i, j] = 1.0
                A[j, i] = 1.0

    # Avoid isolated nodes to keep B and B_star well behaved.
    deg = A.sum(axis=1)
    for i in np.where(deg == 0)[0]:
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        A[i, j] = 1.0
        A[j, i] = 1.0

    D = np.diag(A.sum(axis=1))
    return {
        "n": n,
        "A": A,
        "D": D,
        "group": group,
    }


def sample_theta_star(group, scenario, rng):
    """Group-based controlled theta*: theta_i^* = mu_g + eps_i."""
    mu_map = {
        "S": scenario["mu_S"],
        "W": scenario["mu_W"],
        "N": scenario["mu_N"],
    }
    means = np.array([mu_map[g] for g in group], dtype=float)
    eps = rng.normal(0, scenario["sigma_theta"], size=group.shape[0])
    return means + eps


def generate_labels(theta_star, alpha_true, rng):
    """
    BSS-faithful observation model:
      pi_i^* = (1-alpha*) * sigmoid(theta_i^*)
      Y_i ~ Bernoulli(pi_i^*)
    """
    pi_star = (1.0 - alpha_true) * inv_logit(theta_star)
    pi_star = np.clip(pi_star, 1e-10, 1 - 1e-10)
    Y = rng.binomial(1, pi_star).astype(float)
    return Y, pi_star


def build_B(graph):
    G = solve(graph["D"], graph["A"])
    return np.eye(graph["n"]) - DAMPING * G.T


def build_B_star(graph):
    d_diag = np.diag(graph["D"]).astype(float)
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d_diag))
    return np.eye(graph["n"]) - DAMPING * d_inv_sqrt @ graph["A"] @ d_inv_sqrt


def bss_initial_state(graph, Y):
    """
    Original R initialization:
      u_0 <- solve(B) %*% rep(1-d, n)
      Base_Line <- solve(B_star) %*% Y
      ini <- Base_to_start(Base_Line)
      alpha_est <- alpha_find(u_0, Y, grid)
    """
    B = build_B(graph)
    B_star = build_B_star(graph)
    n = graph["n"]

    u_0 = solve(B, np.full(n, 1.0 - DAMPING))
    base_line = solve(B_star, Y)
    ini = base_to_start(base_line)
    alpha_est = alpha_find(u_0, Y, GRID)

    return {
        "B": B,
        "B_star": B_star,
        "u_0": u_0,
        "base_line": base_line,
        "ini": ini,
        "alpha_est": alpha_est,
    }


def choose_top_k(theta_star):
    return max(1, int(np.sum(theta_star > 0)))


def topk_overlap(theta_star, theta_hat, k):
    truth_top = set(np.argsort(theta_star)[::-1][:k].tolist())
    est_top = set(np.argsort(theta_hat)[::-1][:k].tolist())
    return len(truth_top & est_top) / max(k, 1)


def ndcg_at_k(theta_star, theta_hat, k):
    # Normalize rank-based relevance to [0, 1] to avoid overflow in gain terms.
    rel = np.argsort(np.argsort(theta_star)).astype(float)
    rel = rel / max(len(theta_star) - 1, 1)
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    ideal_rank = np.argsort(theta_star)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[pred_rank] - 1) * discounts)
    idcg = np.sum((2 ** rel[ideal_rank] - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def summarize_estimates(method, theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall_time_sec):
    if np.std(theta_hat) < 1e-12:
        slope = 0.0
        intercept = float(np.mean(theta_star))
        theta_hat_cal = np.full_like(theta_hat, intercept, dtype=float)
    else:
        slope, intercept = np.polyfit(theta_hat, theta_star, 1)
        theta_hat_cal = intercept + slope * theta_hat
    k = choose_top_k(theta_star)
    return TrialResult(
        method=method,
        mse_theta=float(np.mean((theta_hat - theta_star) ** 2)),
        mse_pi=float(np.mean((pi_hat - pi_star) ** 2)),
        mse_calibrated=float(np.mean((theta_hat_cal - theta_star) ** 2)),
        spearman=float(spearmanr(theta_star, theta_hat).statistic),
        kendall=float(kendalltau(theta_star, theta_hat).statistic),
        topk_overlap=float(topk_overlap(theta_star, theta_hat, k)),
        ndcg_at_k=float(ndcg_at_k(theta_star, theta_hat, k)),
        alpha_hat=float(alpha_hat),
        n_obs=int(np.sum(Y)),
        wall_time_sec=float(wall_time_sec),
        theta_hat=theta_hat,
        pi_hat=pi_hat,
    )


def run_acmh(graph, Y, theta_star, pi_star, init_state):
    """
    acMH comparison arm using the existing BSS Gibbs/MH implementation.
    """
    t0 = time.perf_counter()
    result = gibbs_mh(
        Burn_in=BURN_IN,
        T=T,
        ini=init_state["ini"],
        n=graph["n"],
        graph=graph,
        Y=Y,
        B=init_state["B"],
        u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"],
        grid=GRID,
        verbose=False,
    )
    wall_time_sec = time.perf_counter() - t0
    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    pi_hat = (1.0 - alpha_hat) * inv_logit(theta_hat)
    pi_hat = np.clip(pi_hat, 1e-10, 1 - 1e-10)
    return summarize_estimates("acMH", theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall_time_sec)


def run_awsgld(graph, Y, theta_star, pi_star, init_state):
    """
    AWSGLD arm using keyphrase_functions_awsgld_0422.gibbs_mh (theta update
    replaced by AWSGLD). Same signature and return keys as acMH, so the same
    init_state is reused for a fair comparison.
    """
    t0 = time.perf_counter()
    result = gibbs_mh_awsgld(
        Burn_in=BURN_IN,
        T=T,
        ini=init_state["ini"],
        n=graph["n"],
        graph=graph,
        Y=Y,
        B=init_state["B"],
        u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"],
        grid=GRID,
        verbose=False,
    )
    wall_time_sec = time.perf_counter() - t0
    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    pi_hat = (1.0 - alpha_hat) * inv_logit(theta_hat)
    pi_hat = np.clip(pi_hat, 1e-10, 1 - 1e-10)
    return summarize_estimates("AWSGLD", theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall_time_sec)


def trial_payload(result):
    return {
        "method": result.method,
        "mse_theta": result.mse_theta,
        "mse_pi": result.mse_pi,
        "mse_calibrated": result.mse_calibrated,
        "spearman": result.spearman,
        "kendall": result.kendall,
        "topk_overlap": result.topk_overlap,
        "ndcg_at_k": result.ndcg_at_k,
        "alpha_hat": result.alpha_hat,
        "n_obs": result.n_obs,
        "wall_time_sec": result.wall_time_sec,
    }


def summarize_trials(results):
    metric_names = [
        "mse_theta",
        "mse_pi",
        "mse_calibrated",
        "spearman",
        "kendall",
        "topk_overlap",
        "ndcg_at_k",
        "alpha_hat",
        "n_obs",
        "wall_time_sec",
    ]
    summary = {}
    for metric in metric_names:
        vals = np.array([getattr(r, metric) for r in results], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "se": float(np.std(vals, ddof=0) / np.sqrt(len(vals))),
            "values": [float(v) for v in vals],
        }
    return summary


def plot_method_summary(method_summaries, out_path):
    methods = list(method_summaries.keys())
    x = np.arange(len(methods))

    metrics = [
        ("mse_theta", "MSE of theta", None),
        ("mse_pi", "MSE of pi", None),
        ("spearman", "Spearman", (0, 1)),
        ("ndcg_at_k", "NDCG@k", (0, 1)),
    ]
    colors = {"acMH": "#2F6DB2", "AWSGLD": "#D85A30"}

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.35, wspace=0.28, top=0.92, bottom=0.08)

    for ax, (metric, title, ylim) in zip(axes.flat, metrics):
        means = [method_summaries[m][metric]["mean"] for m in methods]
        ses = [method_summaries[m][metric]["se"] for m in methods]
        ax.bar(
            x,
            means,
            yerr=ses,
            capsize=4,
            color=[colors.get(m, "#888888") for m in methods],
            alpha=0.9,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        if ylim is not None:
            ax.set_ylim(*ylim)

    fig.text(
        0.5,
        0.02,
        "Same synthetic graph, same theta*, same Y, same BSS initialization for both methods",
        ha="center",
        fontsize=10,
        color="gray",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "awsgld_acmh_theta_recovery_summary_v2_sparse.json")
    png_path = os.path.join(out_dir, "awsgld_acmh_theta_recovery_summary_v2_sparse.png")

    print("=" * 72)
    print("Controlled theta/pi recovery simulation: AWSGLD vs acMH")
    print(f"Scenario: {DEFAULT_SCENARIO['name']}")
    print(f"R={R}, T={T}, Burn-in={BURN_IN}")
    print("=" * 72)

    results_by_method = {"acMH": [], "AWSGLD": []}
    aws_placeholder_logged = False

    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        graph = build_block_graph(DEFAULT_SCENARIO, rng)
        theta_star = sample_theta_star(graph["group"], DEFAULT_SCENARIO, rng)
        Y, pi_star = generate_labels(theta_star, DEFAULT_SCENARIO["alpha_true"], rng)
        init_state = bss_initial_state(graph, Y)

        acmh_result = run_acmh(graph, Y, theta_star, pi_star, init_state)
        results_by_method["acMH"].append(acmh_result)
        print(
            f"[Trial {r+1}/{R}] acMH   | n_obs={acmh_result.n_obs} | "
            f"MSE(theta)={acmh_result.mse_theta:.4f} | "
            f"Spearman={acmh_result.spearman:.4f} | "
            f"NDCG@k={acmh_result.ndcg_at_k:.4f} | "
            f"time={acmh_result.wall_time_sec:.2f}s"
        )

        try:
            awsgld_result = run_awsgld(graph, Y, theta_star, pi_star, init_state)
            results_by_method["AWSGLD"].append(awsgld_result)
            print(
                f"[Trial {r+1}/{R}] AWSGLD | n_obs={awsgld_result.n_obs} | "
                f"MSE(theta)={awsgld_result.mse_theta:.4f} | "
                f"Spearman={awsgld_result.spearman:.4f} | "
                f"NDCG@k={awsgld_result.ndcg_at_k:.4f} | "
                f"time={awsgld_result.wall_time_sec:.2f}s"
            )
        except NotImplementedError as exc:
            if not aws_placeholder_logged:
                print(f"AWSGLD pending: {exc}")
                aws_placeholder_logged = True

    summaries = {}
    for method, results in results_by_method.items():
        if not results:
            continue
        summaries[method] = summarize_trials(results)

    if summaries:
        plot_method_summary(summaries, png_path)

    payload = {
        "settings": {
            "scenario": DEFAULT_SCENARIO,
            "block_probs": BLOCK_PROBS,
            "R": R,
            "T": T,
            "BURN_IN": BURN_IN,
            "damping": DAMPING,
            "grid_size": int(len(GRID)),
            "init_rule": {
                "u_0": "solve(B, rep(1-d, n))",
                "Base_Line": "solve(B_star, Y)",
                "ini": "base_to_start(Base_Line)",
                "alpha_est": "alpha_find(u_0, Y, grid)",
            },
        },
        "methods": {
            method: {
                "summary": summary,
                "trials": [trial_payload(r) for r in results_by_method[method]],
            }
            for method, summary in summaries.items()
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("-" * 72)
    for method, summary in summaries.items():
        print(
            f"[{method}] MSE(theta)={summary['mse_theta']['mean']:.4f} "
            f"(SE={summary['mse_theta']['se']:.4f}) | "
            f"MSE(pi)={summary['mse_pi']['mean']:.4f} | "
            f"Spearman={summary['spearman']['mean']:.4f} | "
            f"NDCG@k={summary['ndcg_at_k']['mean']:.4f} | "
            f"time={summary['wall_time_sec']['mean']:.2f}s "
            f"(SE={summary['wall_time_sec']['se']:.2f}s)"
        )
    print(f"Saved JSON    -> {json_path}")
    if summaries:
        print(f"Saved summary -> {png_path}")


if __name__ == "__main__":
    main()
