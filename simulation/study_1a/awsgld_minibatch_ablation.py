"""
Plan A — Minibatch AWSGLD on Easy scenario at large n.

Compares:
  - acMH (baseline, full-batch MH)
  - AWSGLD full-batch (batch_size = n)
  - AWSGLD minibatch with several batch sizes

All methods share the same synthetic graph, theta*, Y, BSS init.
Wall-clock time recorded per trial.

Run
---
python3 simulation/awsgld_minibatch_ablation.py
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
# 이전 파일명: keyphrase_functions_awsgld_minibatch_0504 → 현재는
# keyphrase_functions_awsgld 로 통합됨 (gibbs_mh 가 batch_size 인자 받음).
from keyphrase_functions_awsgld import (  # noqa: E402
    gibbs_mh as gibbs_mh_awsgld_mb,
)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DAMPING = 0.85
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

T = 5000
BURN_IN = 1000
R = 2
SEED_BASE = 20260504

# Easy scenario, scaled to n=1500
BLOCK_PROBS = {
    "within": 0.20,
    "between_sw": 0.03,
    "between_other": 0.005,
}

DEFAULT_SCENARIO = {
    "name": "MinibatchEasy_n1500",
    "n_total": 1500,
    "rho_S": 0.20,
    "rho_W": 0.20,
    "rho_N": 0.60,
    "mu_S": 2.5,
    "mu_W": 1.0,
    "mu_N": -2.5,
    "sigma_theta": 0.35,
    "alpha_true": 0.20,
}

# Methods to run. None batch_size means full-batch.
AWSGLD_BATCH_SIZES = [None, 750, 300, 100]


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

    deg = A.sum(axis=1)
    for i in np.where(deg == 0)[0]:
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        A[i, j] = 1.0
        A[j, i] = 1.0

    D = np.diag(A.sum(axis=1))
    return {"n": n, "A": A, "D": D, "group": group}


def sample_theta_star(group, scenario, rng):
    mu_map = {"S": scenario["mu_S"], "W": scenario["mu_W"], "N": scenario["mu_N"]}
    means = np.array([mu_map[g] for g in group], dtype=float)
    eps = rng.normal(0, scenario["sigma_theta"], size=group.shape[0])
    return means + eps


def generate_labels(theta_star, alpha_true, rng):
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
    B = build_B(graph)
    B_star = build_B_star(graph)
    n = graph["n"]
    u_0 = solve(B, np.full(n, 1.0 - DAMPING))
    base_line = solve(B_star, Y)
    ini = base_to_start(base_line)
    alpha_est = alpha_find(u_0, Y, GRID)
    return {"B": B, "B_star": B_star, "u_0": u_0, "base_line": base_line,
            "ini": ini, "alpha_est": alpha_est}


def choose_top_k(theta_star):
    return max(1, int(np.sum(theta_star > 0)))


def topk_overlap(theta_star, theta_hat, k):
    truth_top = set(np.argsort(theta_star)[::-1][:k].tolist())
    est_top = set(np.argsort(theta_hat)[::-1][:k].tolist())
    return len(truth_top & est_top) / max(k, 1)


def ndcg_at_k(theta_star, theta_hat, k):
    rel = np.argsort(np.argsort(theta_star)).astype(float)
    rel = rel / max(len(theta_star) - 1, 1)
    pred_rank = np.argsort(theta_hat)[::-1][:k]
    ideal_rank = np.argsort(theta_star)[::-1][:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[pred_rank] - 1) * discounts)
    idcg = np.sum((2 ** rel[ideal_rank] - 1) * discounts)
    return float(dcg / idcg) if idcg > 0 else 0.0


def summarize_estimates(method, theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall):
    if np.std(theta_hat) < 1e-12:
        slope, intercept = 0.0, float(np.mean(theta_star))
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
        wall_time_sec=float(wall),
        theta_hat=theta_hat,
        pi_hat=pi_hat,
    )


def run_acmh(graph, Y, theta_star, pi_star, init_state):
    t0 = time.perf_counter()
    result = gibbs_mh(
        Burn_in=BURN_IN, T=T, ini=init_state["ini"], n=graph["n"], graph=graph,
        Y=Y, B=init_state["B"], u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"], grid=GRID, verbose=False,
    )
    wall = time.perf_counter() - t0
    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    pi_hat = (1.0 - alpha_hat) * inv_logit(theta_hat)
    pi_hat = np.clip(pi_hat, 1e-10, 1 - 1e-10)
    return summarize_estimates("acMH", theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall)


def run_awsgld_mb(graph, Y, theta_star, pi_star, init_state, batch_size):
    method_label = "AWSGLD_full" if batch_size is None else f"AWSGLD_b{batch_size}"
    t0 = time.perf_counter()
    result = gibbs_mh_awsgld_mb(
        Burn_in=BURN_IN, T=T, ini=init_state["ini"], n=graph["n"], graph=graph,
        Y=Y, B=init_state["B"], u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"], grid=GRID,
        batch_size=batch_size, verbose=False,
    )
    wall = time.perf_counter() - t0
    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    pi_hat = (1.0 - alpha_hat) * inv_logit(theta_hat)
    pi_hat = np.clip(pi_hat, 1e-10, 1 - 1e-10)
    return summarize_estimates(method_label, theta_star, pi_star, theta_hat, pi_hat,
                               alpha_hat, Y, wall)


def trial_payload(r):
    return {
        "method": r.method, "mse_theta": r.mse_theta, "mse_pi": r.mse_pi,
        "mse_calibrated": r.mse_calibrated, "spearman": r.spearman, "kendall": r.kendall,
        "topk_overlap": r.topk_overlap, "ndcg_at_k": r.ndcg_at_k, "alpha_hat": r.alpha_hat,
        "n_obs": r.n_obs, "wall_time_sec": r.wall_time_sec,
    }


def summarize_trials(results):
    metric_names = ["mse_theta","mse_pi","mse_calibrated","spearman","kendall",
                    "topk_overlap","ndcg_at_k","alpha_hat","n_obs","wall_time_sec"]
    summary = {}
    for metric in metric_names:
        vals = np.array([getattr(r, metric) for r in results], dtype=float)
        summary[metric] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "values": [float(v) for v in vals],
        }
    return summary


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(out_dir, "awsgld_minibatch_ablation_summary.json")
    png_path = os.path.join(out_dir, "awsgld_minibatch_ablation_summary.png")

    print("=" * 72)
    print(f"Plan A — Minibatch AWSGLD on {DEFAULT_SCENARIO['name']}")
    print(f"R={R}, T={T}, Burn-in={BURN_IN}, n={DEFAULT_SCENARIO['n_total']}")
    print(f"Methods: acMH, AWSGLD batch_sizes={AWSGLD_BATCH_SIZES}")
    print("=" * 72)

    method_keys = ["acMH"] + [
        "AWSGLD_full" if b is None else f"AWSGLD_b{b}" for b in AWSGLD_BATCH_SIZES
    ]
    results_by_method = {m: [] for m in method_keys}

    for r in range(R):
        rng = np.random.default_rng(SEED_BASE + r)
        graph = build_block_graph(DEFAULT_SCENARIO, rng)
        theta_star = sample_theta_star(graph["group"], DEFAULT_SCENARIO, rng)
        Y, pi_star = generate_labels(theta_star, DEFAULT_SCENARIO["alpha_true"], rng)
        init_state = bss_initial_state(graph, Y)

        # acMH baseline
        ac = run_acmh(graph, Y, theta_star, pi_star, init_state)
        results_by_method["acMH"].append(ac)
        print(f"[Trial {r+1}/{R}] acMH         | n_obs={ac.n_obs} | "
              f"MSE(θ)={ac.mse_theta:.4f} | Spear={ac.spearman:.4f} | "
              f"NDCG={ac.ndcg_at_k:.4f} | t={ac.wall_time_sec:.1f}s")

        # AWSGLD across batch sizes
        for bs in AWSGLD_BATCH_SIZES:
            res = run_awsgld_mb(graph, Y, theta_star, pi_star, init_state, bs)
            results_by_method[res.method].append(res)
            print(f"[Trial {r+1}/{R}] {res.method:13s}| n_obs={res.n_obs} | "
                  f"MSE(θ)={res.mse_theta:.4f} | Spear={res.spearman:.4f} | "
                  f"NDCG={res.ndcg_at_k:.4f} | t={res.wall_time_sec:.1f}s")

    summaries = {m: summarize_trials(results_by_method[m]) for m in method_keys
                 if results_by_method[m]}

    payload = {
        "settings": {
            "scenario": DEFAULT_SCENARIO,
            "block_probs": BLOCK_PROBS,
            "R": R, "T": T, "BURN_IN": BURN_IN, "damping": DAMPING,
            "grid_size": int(len(GRID)),
            "awsgld_batch_sizes": AWSGLD_BATCH_SIZES,
        },
        "methods": {
            m: {"summary": summaries[m],
                "trials": [trial_payload(r) for r in results_by_method[m]]}
            for m in summaries
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Bar plot — same format as v2 simulations.
    methods = list(summaries.keys())
    x = np.arange(len(methods))
    metrics = [
        ("mse_theta", "MSE(θ)", None),
        ("spearman", "Spearman", (0, 1)),
        ("ndcg_at_k", "NDCG@k", (0, 1)),
        ("wall_time_sec", "Wall-clock (s)", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.10)
    for ax, (metric, title, ylim) in zip(axes.flat, metrics):
        means = [summaries[m][metric]["mean"] for m in methods]
        stds = [summaries[m][metric]["std"] for m in methods]
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=20, ha="right")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.15)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if metric == "wall_time_sec":
            ax.set_yscale("log")
    fig.suptitle(
        f"Plan A — n={DEFAULT_SCENARIO['n_total']}, R={R}, T={T}",
        fontsize=12, fontweight="bold", y=0.98,
    )
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("-" * 72)
    for m in methods:
        s = summaries[m]
        print(f"[{m:13s}] MSE(θ)={s['mse_theta']['mean']:.4f}±{s['mse_theta']['std']:.4f} | "
              f"Spear={s['spearman']['mean']:.4f}±{s['spearman']['std']:.4f} | "
              f"NDCG={s['ndcg_at_k']['mean']:.4f}±{s['ndcg_at_k']['std']:.4f} | "
              f"t={s['wall_time_sec']['mean']:.1f}±{s['wall_time_sec']['std']:.1f}s")
    print(f"Saved JSON    -> {json_path}")
    print(f"Saved summary -> {png_path}")


if __name__ == "__main__":
    main()
