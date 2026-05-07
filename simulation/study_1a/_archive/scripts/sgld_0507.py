"""
SGLD/qSGLD/cycSGLD on Moderate, Difficult, Sparse scenarios (R=1 each).

Mirrors AWSGLD-vs-acMH v2 scenario parameters one-to-one
(see awsgld_acmh_theta_recovery_simulation_0422_v2_*.py).

Run
---
python3 /home/jiyoon/3차/BSS-Keyphrase-Extraction-master/simlation/sgld_0507.py
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None
try:
    from scipy.linalg import solve
    from scipy.stats import spearmanr, kendalltau, invgamma
except ModuleNotFoundError:
    solve = np.linalg.solve

    class invgamma:
        @staticmethod
        def rvs(a, scale=1.0):
            return 1.0 / np.random.gamma(shape=a, scale=1.0 / scale)

    def _rankdata(x):
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(len(x), dtype=float)
        return ranks

    def spearmanr(x, y):
        rx = _rankdata(np.asarray(x))
        ry = _rankdata(np.asarray(y))
        if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
            return SimpleNamespace(statistic=np.nan)
        return SimpleNamespace(statistic=float(np.corrcoef(rx, ry)[0, 1]))

    def kendalltau(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        concordant = 0
        discordant = 0
        for i in range(len(x) - 1):
            prod = np.sign(x[i + 1:] - x[i]) * np.sign(y[i + 1:] - y[i])
            concordant += int(np.sum(prod > 0))
            discordant += int(np.sum(prod < 0))
        denom = concordant + discordant
        value = (concordant - discordant) / denom if denom > 0 else np.nan
        return SimpleNamespace(statistic=float(value))


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
DAMPING = 0.85
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

T = 5000
BURN_IN = 1000
R = 1
SEED_BASE = 20260507

SGLD_TAU = 1.0
SGLD_LR_BASE = 0.02
QSGLD_LR_BASE = 0.3
CYCSGLD_LR_BASE = 0.01
CYCSGLD_CYCLES = 10

# Per-scenario settings replicate the matching v2 simulation files.
SCENARIOS = [
    {
        "scenario": {
            "name": "ControlledModerate_v2",
            "n_total": 400,
            "rho_S": 0.20,
            "rho_W": 0.20,
            "rho_N": 0.60,
            "mu_S": 2.0,
            "mu_W": 0.5,
            "mu_N": -1.8,
            "sigma_theta": 0.5,
            "alpha_true": 0.35,
        },
        "block_probs": {
            "within": 0.20,
            "between_sw": 0.03,
            "between_other": 0.005,
        },
    },
    {
        "scenario": {
            "name": "ControlledDifficult_v2",
            "n_total": 400,
            "rho_S": 0.20,
            "rho_W": 0.20,
            "rho_N": 0.60,
            "mu_S": 1.5,
            "mu_W": 0.0,
            "mu_N": -1.0,
            "sigma_theta": 0.6,
            "alpha_true": 0.50,
        },
        "block_probs": {
            "within": 0.15,
            "between_sw": 0.05,
            "between_other": 0.010,
        },
    },
    {
        "scenario": {
            "name": "ControlledSparse_v2_OptB",
            "n_total": 400,
            "rho_S": 0.10,
            "rho_W": 0.18,
            "rho_N": 0.72,
            "mu_S": 2.0,
            "mu_W": 1.0,
            "mu_N": -1.0,
            "sigma_theta": 0.55,
            "alpha_true": 0.40,
        },
        "block_probs": {
            "within": 0.20,
            "between_sw": 0.03,
            "between_other": 0.005,
        },
    },
]

# Active scenario settings — overwritten per loop in main().
BLOCK_PROBS = SCENARIOS[0]["block_probs"]
DEFAULT_SCENARIO = SCENARIOS[0]["scenario"]


def inv_logit(x):
    x = np.clip(x, -700, 700)
    return np.exp(x) / (1.0 + np.exp(x))


def alpha_lk(base_line, Y, alpha):
    pi = inv_logit(base_line)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = (1.0 - alpha) * pi
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    return np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp))


def alpha_find(base_line, Y, grid):
    return grid[np.argmax([alpha_lk(base_line, Y, alpha) for alpha in grid])]


def base_to_start(base_line):
    ini_point = base_line.copy()
    ini_point[ini_point >= 1] = 0.99
    ini_point[ini_point <= 0] = 0.01
    return np.log(ini_point / (1.0 - ini_point))


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


def grad_posterior_energy_fixed_btb(Y, alpha, theta, u_0, B, sigma2, BtB):
    """Gradient of energy = -gradient of log posterior, with precomputed B.T @ B."""
    pi_theta = inv_logit(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)

    dpi_dtheta = pi_theta * (1.0 - pi_theta)
    temp = (1.0 - alpha) * pi_theta
    temp = np.clip(temp, 1e-10, 1 - 1e-10)
    denom = np.clip(1.0 - temp, 1e-10, None)

    grad_ll = np.zeros_like(theta)
    seed_mask = Y == 1
    unlabeled_mask = ~seed_mask

    if np.any(seed_mask):
        grad_ll[seed_mask] = 1.0 - pi_theta[seed_mask]
    if np.any(unlabeled_mask):
        grad_ll[unlabeled_mask] = (
            -(1.0 - alpha) * dpi_dtheta[unlabeled_mask] / denom[unlabeled_mask]
        )

    grad_prior = -BtB @ (theta - u_0) / sigma2
    return -(grad_ll + grad_prior)


def _preconditioner(B, n):
    BtB = B.T @ B
    ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n))
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    return BtB, P, L


def _sgld_result(theta_store, alpha_store):
    prob = inv_logit(theta_store)
    return {
        "poster_pi_md": np.median(prob[BURN_IN:T, :], axis=0),
        "poster_pi_mn": np.mean(prob[BURN_IN:T, :], axis=0),
        "theta_store": theta_store,
        "alpha_mn": np.mean(alpha_store[BURN_IN:]),
        "alpha_md": np.median(alpha_store[BURN_IN:]),
        "accept": T,
    }


def run_sgld_variant(method, graph, Y, theta_star, pi_star, init_state):
    """
    SGLD-family comparison arm.

    method='SGLD'    : vanilla SGLD update.
    method='qSGLD'   : quasi/preconditioned SGLD using fixed graph geometry.
    method='cycSGLD' : vanilla SGLD with cyclical learning rate/temperature.
    """
    t0 = time.perf_counter()
    n = graph["n"]
    B = init_state["B"]
    u_0 = init_state["u_0"]
    theta = init_state["ini"].copy()
    alpha_est = init_state["alpha_est"]
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    sigma2_floor = 0.5
    BtB, P_precond, L_precond = _preconditioner(B, n)

    for t in range(T):
        C = (B @ (theta - u_0)).T @ (B @ (theta - u_0))
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, sigma2_floor)
        grad_U = grad_posterior_energy_fixed_btb(Y, alpha_est, theta, u_0, B, sigma2, BtB)

        if method == "SGLD":
            eps_k = SGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = theta - eps_k * grad_U + np.sqrt(2.0 * SGLD_TAU * eps_k) * np.random.randn(n)
        elif method == "qSGLD":
            eps_k = QSGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = (
                theta
                - eps_k * (P_precond @ grad_U)
                + np.sqrt(2.0 * SGLD_TAU * eps_k) * (L_precond @ np.random.randn(n))
            )
        elif method == "cycSGLD":
            cycle_len = max(1, T // CYCSGLD_CYCLES)
            cur_beta = (t % cycle_len) / cycle_len
            eps_k = CYCSGLD_LR_BASE / 2.0 * (np.cos(np.pi * min(cur_beta, 0.8)) + 1.0)
            tau_k = SGLD_TAU if cur_beta >= 0.8 else SGLD_TAU / 1e4
            theta = theta - eps_k * grad_U + np.sqrt(2.0 * tau_k * eps_k) * np.random.randn(n)
        else:
            raise ValueError(f"Unknown SGLD method: {method}")

        theta = np.clip(theta, -700, 700)
        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, GRID)
        alpha_store[t] = alpha_est

    result = _sgld_result(theta_store, alpha_store)
    wall_time_sec = time.perf_counter() - t0
    theta_hat = np.mean(result["theta_store"][BURN_IN:, :], axis=0)
    alpha_hat = float(result["alpha_mn"])
    pi_hat = (1.0 - alpha_hat) * inv_logit(theta_hat)
    pi_hat = np.clip(pi_hat, 1e-10, 1 - 1e-10)
    return summarize_estimates(method, theta_star, pi_star, theta_hat, pi_hat, alpha_hat, Y, wall_time_sec)


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
    if plt is None:
        return False

    methods = list(method_summaries.keys())
    x = np.arange(len(methods))

    metrics = [
        ("mse_theta", "MSE of theta", None),
        ("mse_pi", "MSE of pi", None),
        ("spearman", "Spearman", (0, 1)),
        ("ndcg_at_k", "NDCG@k", (0, 1)),
    ]
    colors = {"SGLD": "#2F6DB2", "qSGLD": "#D85A30", "cycSGLD": "#4E9A51"}

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
        "Same synthetic graph, same theta*, same Y, same BSS initialization for all methods",
        ha="center",
        fontsize=10,
        color="gray",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    global BLOCK_PROBS, DEFAULT_SCENARIO

    out_dir = os.path.dirname(os.path.abspath(__file__))
    combined_json_path = os.path.join(out_dir, "sgld_0507_summary.json")

    methods = ["SGLD", "qSGLD", "cycSGLD"]
    combined_payload = {
        "settings_common": {
            "R": R, "T": T, "BURN_IN": BURN_IN, "damping": DAMPING,
            "grid_size": int(len(GRID)),
            "sgld_tau": SGLD_TAU,
            "sgld_lr_base": SGLD_LR_BASE,
            "qsgld_lr_base": QSGLD_LR_BASE,
            "cycsgld_lr_base": CYCSGLD_LR_BASE,
            "cycsgld_cycles": CYCSGLD_CYCLES,
        },
        "scenarios": {},
    }

    for sc_cfg in SCENARIOS:
        BLOCK_PROBS = sc_cfg["block_probs"]
        DEFAULT_SCENARIO = sc_cfg["scenario"]
        scen_name = DEFAULT_SCENARIO["name"]

        print("=" * 72)
        print(f"SGLD/qSGLD/cycSGLD on {scen_name}")
        print(f"R={R}, T={T}, Burn-in={BURN_IN}, n={DEFAULT_SCENARIO['n_total']}")
        print("=" * 72)

        results_by_method = {method: [] for method in methods}

        for r in range(R):
            rng = np.random.default_rng(SEED_BASE + r)
            np.random.seed(SEED_BASE + r)
            graph = build_block_graph(DEFAULT_SCENARIO, rng)
            theta_star = sample_theta_star(graph["group"], DEFAULT_SCENARIO, rng)
            Y, pi_star = generate_labels(theta_star, DEFAULT_SCENARIO["alpha_true"], rng)
            init_state = bss_initial_state(graph, Y)

            for method in methods:
                result = run_sgld_variant(method, graph, Y, theta_star, pi_star, init_state)
                results_by_method[method].append(result)
                print(
                    f"[{scen_name} | Trial {r+1}/{R}] {method:<7} | n_obs={result.n_obs} | "
                    f"MSE(theta)={result.mse_theta:.4f} | "
                    f"Spearman={result.spearman:.4f} | "
                    f"NDCG@k={result.ndcg_at_k:.4f} | "
                    f"time={result.wall_time_sec:.2f}s"
                )

        summaries = {m: summarize_trials(rs) for m, rs in results_by_method.items() if rs}

        png_path = os.path.join(out_dir, f"sgld_0507_{scen_name}.png")
        wrote_plot = False
        if summaries:
            wrote_plot = plot_method_summary(summaries, png_path)

        scen_payload = {
            "settings": {
                "scenario": DEFAULT_SCENARIO,
                "block_probs": BLOCK_PROBS,
            },
            "methods": {
                method: {
                    "summary": summary,
                    "trials": [trial_payload(r) for r in results_by_method[method]],
                }
                for method, summary in summaries.items()
            },
        }
        combined_payload["scenarios"][scen_name] = scen_payload

        print("-" * 72)
        for method, summary in summaries.items():
            print(
                f"[{scen_name}/{method}] "
                f"MSE(θ)={summary['mse_theta']['mean']:.4f} | "
                f"MSE(π)={summary['mse_pi']['mean']:.4f} | "
                f"Spear={summary['spearman']['mean']:.4f} | "
                f"NDCG={summary['ndcg_at_k']['mean']:.4f} | "
                f"time={summary['wall_time_sec']['mean']:.2f}s"
            )
        if wrote_plot:
            print(f"Saved plot -> {png_path}")
        print()

    with open(combined_json_path, "w", encoding="utf-8") as f:
        json.dump(combined_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved combined JSON -> {combined_json_path}")


if __name__ == "__main__":
    main()
