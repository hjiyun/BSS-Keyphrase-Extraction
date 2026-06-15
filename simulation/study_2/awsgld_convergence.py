"""
AWSGLD convergence diagnostics on the Easy scenario (study_1a setting).

SGMCMC 논문 표준 형태로 4-panel diagnostic plot 생성:
    (a) theta trace: 대표 component 9개 (S/W/N 그룹별 3개씩)
    (b) ||theta_k - theta_bar||_2 trace: 벡터 전체 fluctuation
    (c) U(x_k) energy trace: AWSGLD vs Gibbs-MH(acMH) 비교 (multimodal exploration 증거)
    (d) ||theta_bar_k - theta*||^2: running posterior mean MSE (sampler 비교)

burn-in 위치는 점선으로 표시.
"""

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.linalg import solve


# ---------------------------------------------------------------------
# Paths — reuse study_1a's keyphrase function modules.
# ---------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
ORIG_DIR = os.path.join(CODE_DIR, "original")
for _p in (CODE_DIR, ORIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import keyphrase_functions_awsgld as kfa  # noqa: E402


# ---------------------------------------------------------------------
# Settings — match study_1a scenarios (langevin_methods_comparison.py).
# ---------------------------------------------------------------------
DAMPING = 0.85
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

T = 5000
BURN_IN = 1000
SEED = 20260507
BATCH_SIZE = 100

# SGHMC (Chen et al. 2014) — momentum + friction overlay arm for the same
# energy/gradient and sigma2-Gibbs pipeline as AWSGLD.
SGHMC_LR_BASE = 0.01
SGHMC_FRICTION = 0.1
SGHMC_TAU = 1.0
SGHMC_SIGMA2_FLOOR = 0.5
SGHMC_COLOR = "#16A085"

# Each entry mirrors study_1a one-to-one (scenario params + block_probs).
SCENARIOS = {
    "easy": {
        "scenario": {
            "name": "ControlledEasy_v2",
            "n_total": 800, "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
            "mu_S": 2.5, "mu_W": 1.0, "mu_N": -2.5,
            "sigma_theta": 0.35, "alpha_true": 0.20,
        },
        "block_probs": {"within": 0.20, "between_sw": 0.03, "between_other": 0.005},
    },
    "moderate": {
        "scenario": {
            "name": "ControlledModerate_v2",
            "n_total": 800, "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
            "mu_S": 2.0, "mu_W": 0.5, "mu_N": -1.8,
            "sigma_theta": 0.5, "alpha_true": 0.35,
        },
        "block_probs": {"within": 0.20, "between_sw": 0.03, "between_other": 0.005},
    },
    "difficult": {
        "scenario": {
            "name": "ControlledDifficult_v2",
            "n_total": 800, "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
            "mu_S": 1.5, "mu_W": 0.0, "mu_N": -1.0,
            "sigma_theta": 0.6, "alpha_true": 0.50,
        },
        "block_probs": {"within": 0.15, "between_sw": 0.05, "between_other": 0.010},
    },
}

# Active block_probs — set per scenario in main(); build_block_graph reads this global.
BLOCK_PROBS = SCENARIOS["easy"]["block_probs"]


# ---------------------------------------------------------------------
# Helpers (mirror study_1a/langevin_methods_comparison.py).
# ---------------------------------------------------------------------
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
    return {"B": B, "B_star": B_star, "u_0": u_0, "ini": ini, "alpha_est": alpha_est}


# ---------------------------------------------------------------------
# Energy trace utility.
# ---------------------------------------------------------------------
def compute_energy_trace(theta_store, Y, alpha_store, u_0, B, sigma2_store):
    """U(x_k) = -log posterior at each iteration."""
    T_, _ = theta_store.shape
    U = np.zeros(T_)
    for t in range(T_):
        U[t] = kfa.posterior_energy(
            Y, alpha_store[t], theta_store[t], u_0, B, sigma2_store[t]
        )
    return U


# ---------------------------------------------------------------------
# AWSGLD wrapper using the existing module (returns sigma2_store too).
# ---------------------------------------------------------------------
def run_awsgld_with_traces(graph, Y, init_state):
    np.random.seed(SEED)
    res = kfa.gibbs_mh(
        Burn_in=BURN_IN, T=T, ini=init_state["ini"], n=graph["n"], graph=graph,
        Y=Y, B=init_state["B"], u_0=init_state["u_0"],
        alpha_est=init_state["alpha_est"], grid=GRID,
        batch_size=BATCH_SIZE, verbose=False,
    )
    # AWSGLD module stores sigma2_store but not alpha trajectory in the same form;
    # recompute alpha trace by re-running alpha_find on each theta_store row.
    theta_store = res["theta_store"]
    alpha_store = np.array([alpha_find(theta_store[t], Y, GRID) for t in range(T)])
    return {
        "theta_store": theta_store,
        "sigma2_store": res["sigma2_store"],
        "alpha_store": alpha_store,
    }


# ---------------------------------------------------------------------
# SGHMC sampler with traces (same energy/gradient as AWSGLD module).
# ---------------------------------------------------------------------
def run_sghmc_with_traces(graph, Y, init_state):
    """
    SGHMC (Chen et al. 2014) on the same posterior as AWSGLD: auxiliary
    momentum v with a friction term that offsets the minibatch-gradient noise.

    이산화 (mass M=I, B_hat=0):
        theta_{t+1} = theta_t + v_t
        v_{t+1}     = (1 - alpha) v_t - eta * grad_U(theta_t) + N(0, 2 alpha eta tau)
    Returns the same trace structure as run_awsgld_with_traces so it can be
    overlaid on the convergence panels.
    """
    np.random.seed(SEED)
    n = graph["n"]
    B = init_state["B"]
    u_0 = init_state["u_0"]
    theta = init_state["ini"].copy()
    alpha_est = init_state["alpha_est"]
    BtB = B.T @ B
    theta_store = np.zeros((T, n))
    sigma2_store = np.zeros(T)
    alpha_store = np.zeros(T)
    v = np.zeros(n)

    for t in range(T):
        Bv = B @ (theta - u_0)
        C = Bv @ Bv
        # Inverse-gamma Gibbs draw for sigma2 (matches AWSGLD module shape/scale).
        sigma2 = 1.0 / np.random.gamma(n / 2 + 0.001, 1.0 / (C / 2 + 0.001))
        sigma2 = max(sigma2, SGHMC_SIGMA2_FLOOR)
        batch_idx = (np.random.choice(n, size=BATCH_SIZE, replace=False)
                     if BATCH_SIZE is not None and BATCH_SIZE < n else None)
        grad_U = kfa.grad_posterior_energy(
            Y, alpha_est, theta, u_0, B, sigma2, batch_idx=batch_idx, BtB=BtB
        )
        eta_k = SGHMC_LR_BASE / ((t + 1) ** 0.6 + 10.0)
        theta = theta + v
        v = ((1.0 - SGHMC_FRICTION) * v
             - eta_k * grad_U
             + np.sqrt(2.0 * SGHMC_FRICTION * eta_k * SGHMC_TAU) * np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        theta_store[t] = theta
        sigma2_store[t] = sigma2
        alpha_est = alpha_find(theta, Y, GRID)
        alpha_store[t] = alpha_est

    return {
        "theta_store": theta_store,
        "sigma2_store": sigma2_store,
        "alpha_store": alpha_store,
    }


# ---------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------
def pick_representative_components(group, rng, per_group=3):
    """S/W/N 그룹별로 per_group개씩 component 인덱스 선택."""
    chosen = []
    for g in ["S", "W", "N"]:
        idx = np.where(group == g)[0]
        chosen.extend(rng.choice(idx, size=per_group, replace=False).tolist())
    return np.array(chosen)


def running_mean(arr):
    """arr shape (T, n) → running mean along axis 0 at each step k."""
    cum = np.cumsum(arr, axis=0)
    denom = np.arange(1, arr.shape[0] + 1)[:, None]
    return cum / denom


def plot_convergence(awsgld_res, theta_star, group, B, Y, u_0, out_path, scenario,
                     sghmc_res=None):
    rng_plot = np.random.default_rng(SEED + 7)
    comp_idx = pick_representative_components(group, rng_plot, per_group=3)

    theta_aw = awsgld_res["theta_store"]
    theta_sg = sghmc_res["theta_store"] if sghmc_res is not None else None

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.subplots_adjust(hspace=0.32, wspace=0.27, top=0.93, bottom=0.07,
                        left=0.07, right=0.98)

    iters = np.arange(T)

    # ---- (a) theta trace for representative components -----------------
    ax = axes[0, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, len(comp_idx)))
    for c, i in zip(colors, comp_idx):
        ax.plot(iters, theta_aw[:, i], lw=0.6, color=c,
                label=f"i={i} ({group[i]})")
    ax.axvline(BURN_IN, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\theta_k^{(i)}$")
    ax.set_title("(a) AWSGLD: representative component traces", fontsize=11, fontweight="bold")
    ax.legend(fontsize=7, ncol=3, loc="lower right", framealpha=0.85)
    ax.grid(True, alpha=0.2)

    # ---- (b) ||theta_k - theta_bar||_2 trace ---------------------------
    ax = axes[0, 1]
    theta_bar_aw = theta_aw[BURN_IN:].mean(axis=0)
    dist_aw = np.linalg.norm(theta_aw - theta_bar_aw, axis=1)
    ax.plot(iters, dist_aw, color="#9B59B6", lw=0.7, label="AWSGLD")
    if theta_sg is not None:
        theta_bar_sg = theta_sg[BURN_IN:].mean(axis=0)
        dist_sg = np.linalg.norm(theta_sg - theta_bar_sg, axis=1)
        ax.plot(iters, dist_sg, color=SGHMC_COLOR, lw=0.7, label="SGHMC")
    ax.axvline(BURN_IN, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|\theta_k - \bar\theta\|_2$")
    ax.set_title(r"(b) $\|\theta_k - \bar\theta\|_2$ trace",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ---- (c) U(x_k) energy trace ---------------------------------------
    ax = axes[1, 0]
    U_aw = compute_energy_trace(
        theta_aw, Y, awsgld_res["alpha_store"], u_0, B,
        sigma2_store=awsgld_res["sigma2_store"],
    )
    ax.plot(iters, U_aw, color="#9B59B6", lw=0.6, alpha=0.85, label="AWSGLD")
    if theta_sg is not None:
        U_sg = compute_energy_trace(
            theta_sg, Y, sghmc_res["alpha_store"], u_0, B,
            sigma2_store=sghmc_res["sigma2_store"],
        )
        ax.plot(iters, U_sg, color=SGHMC_COLOR, lw=0.6, alpha=0.85, label="SGHMC")
    ax.axvline(BURN_IN, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$U(x_k)$")
    ax.set_title(r"(c) energy trace $U(x_k)$",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ---- (d) ||theta_bar_k - theta*||^2 running posterior mean MSE ----
    ax = axes[1, 1]
    mean_aw = running_mean(theta_aw)
    mse_aw = np.mean((mean_aw - theta_star[None, :]) ** 2, axis=1)
    ax.plot(iters, mse_aw, color="#9B59B6", lw=1.0, label="AWSGLD")
    if theta_sg is not None:
        mean_sg = running_mean(theta_sg)
        mse_sg = np.mean((mean_sg - theta_star[None, :]) ** 2, axis=1)
        ax.plot(iters, mse_sg, color=SGHMC_COLOR, lw=1.0, label="SGHMC")
    ax.axvline(BURN_IN, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("iteration k")
    ax.set_ylabel(r"$\|\bar\theta_k - \theta^*\|^2 / n$")
    ax.set_title(r"(d) running posterior mean MSE vs $\theta^*$",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.2, which="both")

    scen_label = scenario["name"].replace("Controlled", "").replace("_v2", "")
    fig.suptitle(
        f"AWSGLD convergence diagnostics — {scen_label} scenario "
        f"(n={scenario['n_total']}, T={T}, burn-in={BURN_IN})",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ---------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------
def run_one_scenario(key, out_dir):
    global BLOCK_PROBS
    cfg = SCENARIOS[key]
    scenario = cfg["scenario"]
    BLOCK_PROBS = cfg["block_probs"]
    out_path = os.path.join(out_dir, f"awsgld_convergence_{key}.png")

    print("=" * 72)
    print(f"{key} ({scenario['name']}) convergence diagnostics | T={T}, burn-in={BURN_IN}")
    print("=" * 72)

    rng = np.random.default_rng(SEED)
    np.random.seed(SEED)
    graph = build_block_graph(scenario, rng)
    theta_star = sample_theta_star(graph["group"], scenario, rng)
    Y, _ = generate_labels(theta_star, scenario["alpha_true"], rng)
    init_state = bss_initial_state(graph, Y)
    print(f"n_obs (Y=1) = {int(Y.sum())} / {graph['n']}")

    t0 = time.perf_counter()
    awsgld_res = run_awsgld_with_traces(graph, Y, init_state)
    print(f"AWSGLD done in {time.perf_counter() - t0:.1f}s")

    t0 = time.perf_counter()
    sghmc_res = run_sghmc_with_traces(graph, Y, init_state)
    print(f"SGHMC done in {time.perf_counter() - t0:.1f}s")

    plot_convergence(
        awsgld_res, theta_star, graph["group"],
        init_state["B"], Y, init_state["u_0"], out_path, scenario,
        sghmc_res=sghmc_res,
    )


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    # CLI: pick scenarios by name; default = all three.
    args = [a.lower() for a in sys.argv[1:]]
    keys = args if args else ["easy", "moderate", "difficult"]
    for key in keys:
        if key not in SCENARIOS:
            print(f"[skip] unknown scenario '{key}' (choices: {list(SCENARIOS)})")
            continue
        run_one_scenario(key, out_dir)


if __name__ == "__main__":
    main()
