"""
Study 1B — SGLD / qSGLD / cycSGLD 만 따로 비교 (study_1a 와 동일 설정).

배경
----
- 데이터: study_1b/data_seed0.npz (acmh_vs_awsgld.py 와 동일).
- 샘플러: study_1a/langevin_methods_comparison.py 의 SGLD 계열 3종을 그대로 차용.
  - SGLD     : vanilla SGLD update
  - qSGLD    : preconditioned SGLD (P=(B^T B)^{-1}) + Cholesky noise
  - cycSGLD  : cyclical lr + 2-stage tau

설정 (acmh_vs_awsgld.py 와 동일)
-------------------------------
- T              = 5000
- BURN_IN        = 500
- NUM_CHAINS     = 3
- BATCH_SIZE     = 100  (n=400 의 25%)
- Bad init       : 모든 노드 θ^(0) = mu_N (chain 0)
- Dispersed init : chain 1 = mu_W 근처, chain 2 = mu_S 근처

지표
----
acmh_vs_awsgld.py 와 동일 항목:
- Recovery  : MSE all/S/W/N, Spearman, mean(θ̂_g)
- Trap escape: escape time per group, mode visit count
- R-hat (Gelman-Rubin)

출력
----
- sgld_results.npz        : 모든 chain 의 theta_store, theta_hat, R-hat per node
- sgld_metric_summary.json
- sgld_recovery.png       : 4 panel (parity x2 + group MSE bar + group mean bar)
- sgld_trace_<method>.png
- sgld_mode_visit.png

실행
----
python3 simulation/study_1b/sgld_only.py            # seed=0
python3 simulation/study_1b/sgld_only.py 2          # seed=2
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr, invgamma

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


# ──────────────────────────────────────────────────────────────────────────
# 설정 (acmh_vs_awsgld.py 와 매칭)
# ──────────────────────────────────────────────────────────────────────────
T = 5000
BURN_IN = 500
NUM_CHAINS = 3
BATCH_SIZE = 100
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

# SGLD-family 하이퍼파라미터 (study_1a 와 동일)
SGLD_TAU = 1.0
SGLD_LR_BASE = 0.02
QSGLD_LR_BASE = 0.3
CYCSGLD_LR_BASE = 0.01
CYCSGLD_CYCLES = 10
SIGMA2_FLOOR = 0.5

# CLAUDE.md 색상 규칙
C_S, C_W, C_N = "#2F6DB2", "#D85A30", "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}
METHOD_COLOR = {"SGLD": "#2F6DB2", "qSGLD": "#D85A30", "cycSGLD": "#4E9A51"}
METHODS = ("SGLD", "qSGLD", "cycSGLD")


# ──────────────────────────────────────────────────────────────────────────
# Loading
# ──────────────────────────────────────────────────────────────────────────
def load_data(seed):
    path = os.path.join(_THIS_DIR, f"data_seed{seed}.npz")
    d = np.load(path)
    return {
        "theta_star": d["theta_star"], "Y": d["Y"],
        "z": np.array([str(x) for x in d["z"]]),
        "A": d["A"], "B": d["B"], "u_0": d["u_0"],
        "n": int(d["n_total"]),
        "seed": int(d["seed"]),
    }


def alpha_lk(theta, Y, alpha):
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    return float(np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp)))


def alpha_find(theta, Y, grid):
    return float(grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))])


# ──────────────────────────────────────────────────────────────────────────
# SGLD 계열 sampler (study_1a/langevin_methods_comparison.py 와 동일 logic)
# ──────────────────────────────────────────────────────────────────────────
def grad_posterior_energy_fixed_btb(Y, alpha, theta, u_0, sigma2, BtB,
                                    batch_idx=None):
    n = theta.shape[0]
    pi_theta = sigmoid(theta)
    pi_theta = np.clip(pi_theta, 1e-10, 1 - 1e-10)
    dpi_dtheta = pi_theta * (1.0 - pi_theta)
    temp = np.clip((1.0 - alpha) * pi_theta, 1e-10, 1 - 1e-10)
    denom = np.clip(1.0 - temp, 1e-10, None)

    grad_ll = np.zeros_like(theta)
    if batch_idx is None:
        seed_mask = Y == 1
        unl_mask = ~seed_mask
        if np.any(seed_mask):
            grad_ll[seed_mask] = 1.0 - pi_theta[seed_mask]
        if np.any(unl_mask):
            grad_ll[unl_mask] = -(1.0 - alpha) * dpi_dtheta[unl_mask] / denom[unl_mask]
    else:
        scale = n / len(batch_idx)
        Y_b = Y[batch_idx]
        seed_b = batch_idx[Y_b == 1]
        unl_b = batch_idx[Y_b == 0]
        if seed_b.size > 0:
            grad_ll[seed_b] = (1.0 - pi_theta[seed_b]) * scale
        if unl_b.size > 0:
            grad_ll[unl_b] = -(1.0 - alpha) * dpi_dtheta[unl_b] / denom[unl_b] * scale

    grad_prior = -BtB @ (theta - u_0) / sigma2
    return -(grad_ll + grad_prior)


def _preconditioner(B, n):
    BtB = B.T @ B
    ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n))
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    return BtB, P, L


def run_sgld_variant(method, data, ini):
    t0 = time.perf_counter()
    n = data["n"]
    B, u_0, Y = data["B"], data["u_0"], data["Y"]
    theta = ini.copy()
    alpha_est = alpha_find(theta, Y, GRID)
    theta_store = np.zeros((T, n))
    alpha_store = np.zeros(T)
    BtB, P_precond, L_precond = _preconditioner(B, n)

    for t in range(T):
        Bv = B @ (theta - u_0)
        C = Bv @ Bv
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, SIGMA2_FLOOR)
        if BATCH_SIZE is None or BATCH_SIZE >= n:
            batch_idx = None
        else:
            batch_idx = np.random.choice(n, size=BATCH_SIZE, replace=False)
        grad_U = grad_posterior_energy_fixed_btb(
            Y, alpha_est, theta, u_0, sigma2, BtB, batch_idx=batch_idx,
        )

        if method == "SGLD":
            eps_k = SGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = (theta - eps_k * grad_U
                     + np.sqrt(2.0 * SGLD_TAU * eps_k) * np.random.randn(n))
        elif method == "qSGLD":
            eps_k = QSGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = (theta
                     - eps_k * (P_precond @ grad_U)
                     + np.sqrt(2.0 * SGLD_TAU * eps_k) * (L_precond @ np.random.randn(n)))
        elif method == "cycSGLD":
            cycle_len = max(1, T // CYCSGLD_CYCLES)
            cur_beta = (t % cycle_len) / cycle_len
            eps_k = CYCSGLD_LR_BASE / 2.0 * (np.cos(np.pi * min(cur_beta, 0.8)) + 1.0)
            tau_k = SGLD_TAU if cur_beta >= 0.8 else SGLD_TAU / 1e4
            theta = (theta - eps_k * grad_U
                     + np.sqrt(2.0 * tau_k * eps_k) * np.random.randn(n))
        else:
            raise ValueError(f"Unknown method: {method}")

        theta = np.clip(theta, -700, 700)
        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, GRID)
        alpha_store[t] = alpha_est

    return {
        "theta_store": theta_store,
        "alpha_mn": float(np.mean(alpha_store[BURN_IN:])),
        "wall_time": time.perf_counter() - t0,
    }


# ──────────────────────────────────────────────────────────────────────────
# Inits (acmh_vs_awsgld.py 와 동일)
# ──────────────────────────────────────────────────────────────────────────
def make_inits(n, num_chains, rng):
    inits = [np.full(n, PARAMS["mu_N"], dtype=float)]
    targets = [PARAMS["mu_W"], PARAMS["mu_S"], 0.0]
    for i in range(num_chains - 1):
        center = targets[i % len(targets)]
        inits.append(center + rng.normal(0.0, 0.3, size=n))
    return inits[:num_chains]


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def recovery_metrics(theta_hat, theta_star, z):
    metrics = {
        "mse_all": float(np.mean((theta_hat - theta_star) ** 2)),
        "spearman": float(spearmanr(theta_star, theta_hat).statistic),
    }
    for g in ("S", "W", "N"):
        m = z == g
        metrics[f"mse_{g}"] = float(np.mean((theta_hat[m] - theta_star[m]) ** 2))
        metrics[f"mean_theta_hat_{g}"] = float(np.mean(theta_hat[m]))
        metrics[f"mean_theta_star_{g}"] = float(np.mean(theta_star[m]))
    return metrics


def escape_time_per_node(theta_store, z, mu_map, window=0.5):
    T_steps, n = theta_store.shape
    out = np.full(n, np.nan)
    for i in range(n):
        target = mu_map[z[i]]
        diff = np.abs(theta_store[:, i] - target)
        idx = np.where(diff <= window)[0]
        if len(idx) > 0:
            out[i] = float(idx[0])
    return out


def mode_visit_counts(theta_store, mu_map, window=0.5):
    post = theta_store[BURN_IN:]
    counts = {}
    in_any = np.zeros_like(post, dtype=bool)
    for g in ("S", "W", "N"):
        mask = np.abs(post - mu_map[g]) <= window
        counts[g] = int(mask.sum())
        in_any |= mask
    counts["other"] = int((~in_any).sum())
    return counts


def gelman_rubin(chains_theta_post):
    M = len(chains_theta_post)
    if M < 2:
        return {"R_hat_median": np.nan, "R_hat_max": np.nan,
                "R_hat_q90": np.nan, "R_hat_per_node": np.array([])}
    L = min(c.shape[0] for c in chains_theta_post)
    arrs = np.stack([c[:L] for c in chains_theta_post], axis=0)
    chain_means = arrs.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B_over_L = ((chain_means - grand_mean) ** 2).sum(axis=0) / (M - 1)
    W = arrs.var(axis=1, ddof=1).mean(axis=0)
    var_hat = (L - 1) / L * W + B_over_L
    R = np.sqrt(np.clip(var_hat / np.maximum(W, 1e-12), 0, None))
    return {
        "R_hat_median": float(np.median(R)),
        "R_hat_max": float(np.max(R)),
        "R_hat_q90": float(np.quantile(R, 0.90)),
        "R_hat_per_node": R,
    }


# ──────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────
def plot_recovery(data, theta_hat_by_method, out_path):
    z = data["z"]
    theta_star = data["theta_star"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # row 0: parity per method
    for col, method in enumerate(METHODS):
        ax = axes[0, col]
        th = theta_hat_by_method[method]
        for g in ("N", "W", "S"):
            m = z == g
            ax.scatter(theta_star[m], th[m], color=GROUP_COLOR[g],
                       label=f"{g} (n={int(m.sum())})", s=16, alpha=0.7)
        lo = min(theta_star.min(), th.min()) - 0.5
        hi = max(theta_star.max(), th.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\theta^*$"); ax.set_ylabel(r"$\hat\theta$")
        ax.set_title(f"{method} — parity (chain 0)",
                     fontweight="bold", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.15)

    # row 1: group MSE bar | group mean vs mu_g | empty
    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}
    groups = ("S", "W", "N")
    x = np.arange(len(groups))
    width = 0.26

    ax = axes[1, 0]
    for i, method in enumerate(METHODS):
        mses = [np.mean((theta_hat_by_method[method][z == g] - theta_star[z == g]) ** 2)
                for g in groups]
        ax.bar(x + (i - 1) * width, mses, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Group MSE")
    ax.set_title("Group 별 MSE", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y"); ax.legend(fontsize=9)

    ax = axes[1, 1]
    for i, method in enumerate(METHODS):
        means = [theta_hat_by_method[method][z == g].mean() for g in groups]
        ax.bar(x + (i - 1) * width, means, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    for j, g in enumerate(groups):
        ax.hlines(mu_map[g], j - 0.5, j + 0.5, colors=GROUP_COLOR[g],
                  linestyles="--", linewidth=1.6,
                  label=(fr"$\mu_g$ truth" if j == 0 else None))
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(r"mean $\hat\theta_g$")
    ax.set_title(r"mean($\hat\theta_g$) vs $\mu_g$", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y"); ax.legend(fontsize=8)

    axes[1, 2].axis("off")

    fig.suptitle(
        f"Study 1B — SGLD-family recovery (seed={data['seed']}, n={data['n']}, "
        f"T={T}, burn={BURN_IN}, bad init=$\\mu_N$)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_trace(theta_store, z, method, out_path, k_per_group=3, seed=0):
    rng = np.random.default_rng(seed + 1234)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}
    for g in ("N", "W", "S"):
        idx_g = np.where(z == g)[0]
        pick = rng.choice(idx_g, size=min(k_per_group, len(idx_g)), replace=False)
        for j, i in enumerate(pick):
            ax.plot(theta_store[:, i], color=GROUP_COLOR[g], alpha=0.65,
                    lw=0.9, label=(f"{g} (node {i})" if j == 0 else None))
        ax.axhline(mu_map[g], color=GROUP_COLOR[g], ls="--", lw=1.2, alpha=0.8)
    ax.axvline(BURN_IN, color="black", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("MCMC step"); ax.set_ylabel(r"$\theta_i^{(t)}$")
    ax.set_title(
        f"{method} — trace from bad init ($\\theta^{{(0)}}=\\mu_N$)",
        fontweight="bold", fontsize=11,
    )
    ax.grid(True, alpha=0.15); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_mode_visit(visit_by_method, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ("S", "W", "N", "other")
    x = np.arange(len(cats))
    width = 0.26
    for i, method in enumerate(METHODS):
        vals = [visit_by_method[method][c] for c in cats]
        total = sum(vals)
        frac = [v / total for v in vals] if total > 0 else vals
        ax.bar(x + (i - 1) * width, frac, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("fraction of post-burn samples in basin")
    ax.set_title("SGLD-family Mode visit count (chain 0, bad init)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y"); ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main(seed=0):
    data = load_data(seed)
    n = data["n"]
    z = data["z"]
    print("=" * 72)
    print(f"Study 1B — SGLD-family on data_seed{seed}.npz")
    print(f"n={n}, T={T}, burn={BURN_IN}, num_chains={NUM_CHAINS}, "
          f"batch_size={BATCH_SIZE}")
    print(f"Groups: S={int((z=='S').sum())}, W={int((z=='W').sum())}, "
          f"N={int((z=='N').sum())}")
    print("=" * 72)

    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}
    rng = np.random.default_rng(seed + 7777)
    inits = make_inits(n, NUM_CHAINS, rng)

    all_chains = {m: [] for m in METHODS}
    wall_times = {m: [] for m in METHODS}

    for method in METHODS:
        print(f"\n── {method} ──")
        for c_idx, ini in enumerate(inits):
            np.random.seed(seed * 1000 + c_idx)
            print(f"  chain {c_idx}: init mean={ini.mean():+.3f} "
                  f"std={ini.std():.3f}", flush=True)
            res = run_sgld_variant(method, data, ini)
            all_chains[method].append(res["theta_store"])
            wall_times[method].append(res["wall_time"])
            print(f"    -> wall {res['wall_time']:.1f}s")

    # ── Recovery ──
    print("\n" + "─" * 72)
    print("A. Recovery metrics (chain 0)")
    print("─" * 72)
    theta_hat_by_method = {}
    recovery_by_method = {}
    for method in METHODS:
        ts = all_chains[method][0]
        theta_hat = ts[BURN_IN:].mean(axis=0)
        theta_hat_by_method[method] = theta_hat
        rec = recovery_metrics(theta_hat, data["theta_star"], z)
        recovery_by_method[method] = rec
        print(f"[{method:<7}] MSE_all={rec['mse_all']:.3f}  "
              f"Spearman={rec['spearman']:.3f}  "
              f"mean(θ̂)=(S {rec['mean_theta_hat_S']:+.2f}, "
              f"W {rec['mean_theta_hat_W']:+.2f}, "
              f"N {rec['mean_theta_hat_N']:+.2f})")

    # ── Trap escape ──
    print("\n" + "─" * 72)
    print("B. Trap escape (chain 0)")
    print("─" * 72)
    escape_by_method = {}
    mode_visit_by_method = {}
    for method in METHODS:
        ts = all_chains[method][0]
        et = escape_time_per_node(ts, z, mu_map, window=0.5)
        escape_by_method[method] = et
        mv = mode_visit_counts(ts, mu_map, window=0.5)
        mode_visit_by_method[method] = mv
        ssum = []
        for g in ("S", "W", "N"):
            sub = et[z == g]
            n_ok = int(np.sum(~np.isnan(sub)))
            n_all = int(np.sum(z == g))
            med_str = (f"{float(np.nanmedian(sub)):.0f}" if n_ok > 0 else "nan")
            ssum.append(f"{g}={n_ok}/{n_all} (med {med_str})")
        print(f"[{method:<7}] escape: " + "  ".join(ssum))

    # ── R-hat ──
    print("\n" + "─" * 72)
    print("C. R-hat")
    print("─" * 72)
    rhat_by_method = {}
    for method in METHODS:
        post_chains = [ts[BURN_IN:] for ts in all_chains[method]]
        rh = gelman_rubin(post_chains)
        rhat_by_method[method] = rh
        print(f"[{method:<7}] R_hat median={rh['R_hat_median']:.3f}  "
              f"q90={rh['R_hat_q90']:.3f}  max={rh['R_hat_max']:.3f}")

    # ── Plots ──
    print("\n" + "─" * 72)
    print("Saving plots...")
    print("─" * 72)
    plot_recovery(data, theta_hat_by_method,
                  os.path.join(_THIS_DIR, "sgld_recovery.png"))
    for method in METHODS:
        plot_trace(all_chains[method][0], z, method,
                   os.path.join(_THIS_DIR, f"sgld_trace_{method}.png"),
                   k_per_group=3, seed=seed)
    plot_mode_visit(mode_visit_by_method,
                    os.path.join(_THIS_DIR, "sgld_mode_visit.png"))

    # ── Save npz / json ──
    npz_path = os.path.join(_THIS_DIR, "sgld_results.npz")
    npz_kwargs = {
        "theta_star": data["theta_star"], "Y": data["Y"], "z": data["z"],
        "seed": np.int64(seed), "T": np.int64(T), "BURN_IN": np.int64(BURN_IN),
        "NUM_CHAINS": np.int64(NUM_CHAINS), "BATCH_SIZE": np.int64(BATCH_SIZE),
    }
    for method in METHODS:
        for c_idx, ts in enumerate(all_chains[method]):
            npz_kwargs[f"{method}_chain{c_idx}_theta_store"] = ts
        npz_kwargs[f"{method}_theta_hat"] = theta_hat_by_method[method]
        npz_kwargs[f"{method}_escape_time"] = escape_by_method[method]
        npz_kwargs[f"{method}_Rhat_per_node"] = rhat_by_method[method]["R_hat_per_node"]
    np.savez(npz_path, **npz_kwargs)
    print(f"Saved -> {npz_path}")

    summary = {
        "settings": {
            "seed": seed, "n": n, "T": T, "BURN_IN": BURN_IN,
            "NUM_CHAINS": NUM_CHAINS, "BATCH_SIZE": BATCH_SIZE,
            "SGLD_TAU": SGLD_TAU,
            "SGLD_LR_BASE": SGLD_LR_BASE,
            "QSGLD_LR_BASE": QSGLD_LR_BASE,
            "CYCSGLD_LR_BASE": CYCSGLD_LR_BASE,
            "CYCSGLD_CYCLES": CYCSGLD_CYCLES,
            "bad_init": "theta^(0) = mu_N (모든 노드, chain 0)",
            "escape_window": 0.5,
        },
        "groups": {g: int((z == g).sum()) for g in ("S", "W", "N")},
        "mu_map": mu_map,
        "recovery": recovery_by_method,
        "mode_visit": mode_visit_by_method,
        "wall_time_sec": {m: wall_times[m] for m in METHODS},
        "R_hat": {m: {k: v for k, v in rhat_by_method[m].items()
                      if k != "R_hat_per_node"}
                  for m in METHODS},
        "escape_summary": {
            method: {
                g: {
                    "n_escaped": int(np.sum(~np.isnan(escape_by_method[method][z == g]))),
                    "n_total": int(np.sum(z == g)),
                    "median_step": (
                        float(np.nanmedian(escape_by_method[method][z == g]))
                        if np.any(~np.isnan(escape_by_method[method][z == g])) else None
                    ),
                }
                for g in ("S", "W", "N")
            }
            for method in METHODS
        },
    }
    json_path = os.path.join(_THIS_DIR, "sgld_metric_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
