"""
Study 1C — 4 sampler 비교 (SGLD / qSGLD / cycSGLD / AWSGLD), 다중 seed.

acMH 제외.  data_generator.py 가 만든 data_n{N}_seed{S}.npz 들을 읽어
4 sampler 를 동일 init 셋으로 비교, 여러 seed 평균.

샘플링 설정 (n 별)
- n=200  : T=5000,  batch=50
- n=1500 : T=10000, batch=200
- 3 chain (bad init θ⁽⁰⁾=μ_N + dispersed μ_W, μ_S)
- AWSGLD σ²_floor = 1.0
- burn-in = T × 10%

평가
- θ̂ 산출: **3 chain pooled mean over post-burn**
  (chain 0 만 쓰지 않음 → cycSGLD 등의 single-chain lucky 효과 제거)
- 다중 seed (기본 0..4) 평균 + 표준편차

지표
- MSE_all, group MSE, Spearman, NDCG@k, mean(θ̂_g)
- Gelman-Rubin R̂ (median, q90, max)
- ESS median, cost-per-ESS
- wall time per chain

실행
- python3 simulation/study_1c/sampler_comparison.py --n 200
- python3 simulation/study_1c/sampler_comparison.py --n 1500
- python3 simulation/study_1c/sampler_comparison.py --n 200 --seeds 0 1 2
"""
import argparse
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
STUDY_1B = os.path.join(os.path.dirname(_THIS_DIR), "study_1b")
for _p in (_THIS_DIR, STUDY_1B, CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from local_trap_landscape import sigmoid  # noqa: E402
from keyphrase_functions_awsgld import gibbs_mh as gibbs_mh_awsgld  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# ──────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────
NUM_CHAINS = 3
BURN_IN_FRAC = 0.10
AWSGLD_SIGMA2_FLOOR = 1.0
SIGMA2_FLOOR_SGLD = 0.5
SGLD_TAU = 1.0
SGLD_LR_BASE = 0.02
QSGLD_LR_BASE = 0.3
CYCSGLD_LR_BASE = 0.01
CYCSGLD_CYCLES = 10
# SGHMC (Chen et al. 2014) — momentum + friction to offset minibatch noise.
SGHMC_LR_BASE = 0.01
SGHMC_FRICTION = 0.1
SGHMC_TAU = 1.0
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)

SCALE_CFG = {
    200:   {"T": 5000,  "batch": 50,   "k_ndcg": 40},
    1500:  {"T": 10000, "batch": 200,  "k_ndcg": 150},
    10000: {"T": 10000, "batch": 1000, "k_ndcg": 1000},
}

METHODS = ("SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD")
METHOD_COLOR = {
    "SGLD": "#2F6DB2", "qSGLD": "#D85A30",
    "cycSGLD": "#4E9A51", "SGHMC": "#16A085", "AWSGLD": "#9B59B6",
}


# ──────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────
def alpha_lk(theta, Y, alpha):
    pi = sigmoid(theta); pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    return float(np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp)))


def alpha_find(theta, Y, grid):
    return float(grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))])


def _preconditioner(B, n):
    BtB = B.T @ B
    ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n))
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    return BtB, P, L


def grad_post_energy(Y, alpha, theta, u_0, sigma2, BtB, batch_idx=None):
    n = theta.shape[0]
    pi = sigmoid(theta); pi = np.clip(pi, 1e-10, 1 - 1e-10)
    dpi = pi * (1.0 - pi)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    denom = np.clip(1.0 - temp, 1e-10, None)
    grad_ll = np.zeros_like(theta)
    if batch_idx is None:
        sm = Y == 1; um = ~sm
        if np.any(sm):
            grad_ll[sm] = 1.0 - pi[sm]
        if np.any(um):
            grad_ll[um] = -(1.0 - alpha) * dpi[um] / denom[um]
    else:
        scale = n / len(batch_idx)
        Yb = Y[batch_idx]
        sb = batch_idx[Yb == 1]; ub = batch_idx[Yb == 0]
        if sb.size > 0:
            grad_ll[sb] = (1.0 - pi[sb]) * scale
        if ub.size > 0:
            grad_ll[ub] = -(1.0 - alpha) * dpi[ub] / denom[ub] * scale
    grad_prior = -BtB @ (theta - u_0) / sigma2
    return -(grad_ll + grad_prior)


# ──────────────────────────────────────────────────────────────────────────
# Samplers
# ──────────────────────────────────────────────────────────────────────────
def run_sgld_variant(method, data, ini, T, batch_size):
    t0 = time.perf_counter()
    n = data["n"]; B = data["B"]; u_0 = data["u_0"]; Y = data["Y"]
    theta = ini.copy(); alpha_est = alpha_find(theta, Y, GRID)
    theta_store = np.zeros((T, n))
    BtB, P_precond, L_precond = _preconditioner(B, n)
    for t in range(T):
        Bv = B @ (theta - u_0); C = Bv @ Bv
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, SIGMA2_FLOOR_SGLD)
        batch_idx = (np.random.choice(n, size=batch_size, replace=False)
                     if batch_size < n else None)
        grad_U = grad_post_energy(Y, alpha_est, theta, u_0, sigma2, BtB,
                                  batch_idx=batch_idx)
        if method == "SGLD":
            eps_k = SGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = (theta - eps_k * grad_U
                     + np.sqrt(2.0 * SGLD_TAU * eps_k) * np.random.randn(n))
        elif method == "qSGLD":
            eps_k = QSGLD_LR_BASE / ((t + 1) ** 0.6 + 10.0)
            theta = (theta
                     - eps_k * (P_precond @ grad_U)
                     + np.sqrt(2.0 * SGLD_TAU * eps_k)
                     * (L_precond @ np.random.randn(n)))
        elif method == "cycSGLD":
            cycle_len = max(1, T // CYCSGLD_CYCLES)
            cur_beta = (t % cycle_len) / cycle_len
            eps_k = CYCSGLD_LR_BASE / 2.0 * (np.cos(np.pi * min(cur_beta, 0.8)) + 1.0)
            tau_k = SGLD_TAU if cur_beta >= 0.8 else SGLD_TAU / 1e4
            theta = (theta - eps_k * grad_U
                     + np.sqrt(2.0 * tau_k * eps_k) * np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, GRID)
    return {"theta_store": theta_store,
            "wall_time": time.perf_counter() - t0}


def run_sghmc_variant(data, ini, T, batch_size):
    """
    SGHMC (Chen et al. 2014) — run_sgld_variant 와 동일 energy/gradient,
    sigma2-Gibbs 파이프라인. theta 업데이트가 보조 운동량 v + 마찰항을 가짐.

    이산화 (mass M=I, B_hat=0):
        theta_{t+1} = theta_t + v_t
        v_{t+1}     = (1 - alpha) v_t - eta * grad_U(theta_t) + N(0, 2 alpha eta tau)
    eta = eps^2 (decayed lr), alpha = eps*C (friction).
    """
    t0 = time.perf_counter()
    n = data["n"]; B = data["B"]; u_0 = data["u_0"]; Y = data["Y"]
    theta = ini.copy(); alpha_est = alpha_find(theta, Y, GRID)
    theta_store = np.zeros((T, n))
    BtB = B.T @ B
    v = np.zeros(n)
    for t in range(T):
        Bv = B @ (theta - u_0); C = Bv @ Bv
        sigma2 = invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001)
        sigma2 = max(sigma2, SIGMA2_FLOOR_SGLD)
        batch_idx = (np.random.choice(n, size=batch_size, replace=False)
                     if batch_size < n else None)
        grad_U = grad_post_energy(Y, alpha_est, theta, u_0, sigma2, BtB,
                                  batch_idx=batch_idx)
        eta_k = SGHMC_LR_BASE / ((t + 1) ** 0.6 + 10.0)
        theta = theta + v
        v = ((1.0 - SGHMC_FRICTION) * v
             - eta_k * grad_U
             + np.sqrt(2.0 * SGHMC_FRICTION * eta_k * SGHMC_TAU) * np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        theta_store[t, :] = theta
        alpha_est = alpha_find(theta, Y, GRID)
    return {"theta_store": theta_store,
            "wall_time": time.perf_counter() - t0}


def run_awsgld(data, ini, T, burn_in, batch_size):
    t0 = time.perf_counter()
    Y = data["Y"]; B = data["B"]; u_0 = data["u_0"]; n = data["n"]
    graph = {"n": n, "A": data["A"], "D": np.diag(data["A"].sum(axis=1))}
    res = gibbs_mh_awsgld(
        Burn_in=burn_in, T=T, ini=ini.copy(), n=n, graph=graph, Y=Y, B=B, u_0=u_0,
        alpha_est=alpha_find(ini, Y, GRID), grid=GRID,
        batch_size=batch_size, sigma2_floor=AWSGLD_SIGMA2_FLOOR, verbose=False,
    )
    return {"theta_store": res["theta_store"],
            "wall_time": time.perf_counter() - t0}


# ──────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────
def gelman_rubin(chains_post):
    M = len(chains_post)
    if M < 2:
        return {"R_hat_median": np.nan, "R_hat_max": np.nan,
                "R_hat_q90": np.nan}
    L = min(c.shape[0] for c in chains_post)
    arrs = np.stack([c[:L] for c in chains_post], axis=0)
    chain_means = arrs.mean(axis=1)
    grand = chain_means.mean(axis=0)
    B_over_L = ((chain_means - grand) ** 2).sum(axis=0) / (M - 1)
    W = arrs.var(axis=1, ddof=1).mean(axis=0)
    var_hat = (L - 1) / L * W + B_over_L
    R = np.sqrt(np.clip(var_hat / np.maximum(W, 1e-12), 0, None))
    return {"R_hat_median": float(np.median(R)),
            "R_hat_max": float(np.max(R)),
            "R_hat_q90": float(np.quantile(R, 0.90))}


def _acf_fft(x):
    x = x - x.mean(); n = len(x)
    f = np.fft.fft(x, n=2 * n)
    acf = np.fft.ifft(f * np.conj(f))[:n].real
    acf /= acf[0] + 1e-30
    return acf


def ess_median(theta_post, max_lag=None):
    L, n = theta_post.shape
    if max_lag is None:
        max_lag = min(L // 3, 1000)
    ess_arr = np.full(n, np.nan)
    for i in range(n):
        x = theta_post[:, i]
        if x.std() < 1e-10:
            ess_arr[i] = float(L); continue
        acf = _acf_fft(x)[:max_lag]
        rho_sum = 0.0
        for k in range(1, max_lag // 2):
            pair = acf[2 * k] + acf[2 * k + 1]
            if pair <= 0:
                break
            rho_sum += pair
        ess_arr[i] = L / (1.0 + 2.0 * rho_sum)
    ess_arr = np.clip(ess_arr, 1.0, L)
    return float(np.median(ess_arr))


def ndcg_at_k(theta_star, theta_hat, k):
    rel = np.argsort(np.argsort(theta_star)).astype(float)
    rel = rel / max(len(theta_star) - 1, 1)
    pred = np.argsort(theta_hat)[::-1][:k]
    ideal = np.argsort(theta_star)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = np.sum((2 ** rel[pred] - 1) * disc)
    idcg = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(dcg / idcg) if idcg > 0 else 0.0


def compute_metrics(theta_hat, theta_star, z, chains_post,
                    wall_time_chain0, k_ndcg):
    out = {
        "mse_all": float(np.mean((theta_hat - theta_star) ** 2)),
        "spearman": float(spearmanr(theta_star, theta_hat).statistic),
        "wall_chain0": float(wall_time_chain0),
    }
    for g in ("S", "W", "N"):
        m = z == g
        out[f"mse_{g}"] = float(np.mean((theta_hat[m] - theta_star[m]) ** 2))
        out[f"mean_{g}"] = float(theta_hat[m].mean())
    out.update(gelman_rubin(chains_post))
    ess_med = ess_median(chains_post[0])
    out["ess_median"] = ess_med
    out["cost_per_ess"] = wall_time_chain0 / max(ess_med, 1e-9)
    out[f"ndcg_at_{k_ndcg}"] = ndcg_at_k(theta_star, theta_hat, k_ndcg)
    return out


# ──────────────────────────────────────────────────────────────────────────
# Plot (mean ± std across seeds)
# ──────────────────────────────────────────────────────────────────────────
def plot_summary_multiseed(stats_by_method, n, k_ndcg, n_seeds, out_path):
    methods = list(stats_by_method.keys())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    panels = [
        ("mse_all", "MSE_all (낮을수록 좋음)", False),
        ("spearman", "Spearman (높을수록 좋음)", False),
        (f"ndcg_at_{k_ndcg}", f"NDCG@{k_ndcg} (높을수록 좋음)", False),
        ("R_hat_max", "R̂ max (낮을수록 좋음)", True),
        ("cost_per_ess", "Cost per ESS (s) (낮을수록 좋음)", True),
        ("wall_chain0", "Wall (s/chain)", True),
    ]
    for ax, (key, title, logy) in zip(axes.flat, panels):
        means = [stats_by_method[m][key]["mean"] for m in methods]
        stds = [stats_by_method[m][key]["std"] for m in methods]
        bars = ax.bar(methods, means, yerr=stds, capsize=5,
                      color=[METHOD_COLOR[m] for m in methods], alpha=0.85,
                      error_kw={"lw": 1.2})
        for b, mu, sd in zip(bars, means, stds):
            ax.text(b.get_x() + b.get_width() / 2, mu,
                    f"{mu:.3f}\n±{sd:.3f}", ha="center", va="bottom",
                    fontsize=7.5)
        ax.set_title(title, fontweight="bold", fontsize=10.5)
        if logy:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.18, axis="y")
        ax.tick_params(axis="x", labelsize=9)

    fig.suptitle(
        f"Study 1C — scale n={n}  (T={SCALE_CFG[n]['T']}, 3 chains pooled, "
        f"{n_seeds} seeds avg, bad init μ_N)",
        fontweight="bold", fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def make_inits(n, num_chains, mu_map, rng):
    """3 chain 모두 Normal(0, 1) 에서 random init (bad init 제거)."""
    return [rng.normal(0.0, 1.0, size=n) for _ in range(num_chains)]


def load_data(n, seed):
    path = os.path.join(_THIS_DIR, f"data_n{n}_seed{seed}.npz")
    if not os.path.exists(path):
        sys.exit(f"Not found: {path}\n  → 먼저 `python3 data_generator.py "
                 f"--n {n} --seed {seed}` 실행")
    d = np.load(path)
    return {
        "theta_star": d["theta_star"], "Y": d["Y"],
        "z": np.array([str(x) for x in d["z"]]),
        "A": d["A"], "B": d["B"], "u_0": d["u_0"],
        "n": int(d["n_total"]), "seed": int(d["seed"]),
        "mu_map": {"S": float(d["param_mu_S"]),
                   "W": float(d["param_mu_W"]),
                   "N": float(d["param_mu_N"])},
    }


def run_one_seed(n, seed, T, BURN_IN, batch, K_NDCG):
    """단일 seed 의 4 sampler × 3 chain 실행 + pooled metrics."""
    data = load_data(n, seed)
    z = data["z"]
    rng = np.random.default_rng(seed + 7777)
    inits = make_inits(n, NUM_CHAINS, data["mu_map"], rng)

    metrics_by_method = {}
    for method in METHODS:
        print(f"  [{method}]", end="", flush=True)
        chains = []
        wall_chain0 = None
        for c_idx, ini in enumerate(inits):
            np.random.seed(seed * 1000 + c_idx)
            if method == "AWSGLD":
                res = run_awsgld(data, ini, T, BURN_IN, batch)
            elif method == "SGHMC":
                res = run_sghmc_variant(data, ini, T, batch)
            else:
                res = run_sgld_variant(method, data, ini, T, batch)
            chains.append(res["theta_store"])
            if c_idx == 0:
                wall_chain0 = res["wall_time"]
            print(f" {res['wall_time']:.0f}s", end="", flush=True)
        # 3 chain pooled posterior mean ← 핵심 변경점
        chains_post = [c[BURN_IN:] for c in chains]
        theta_hat = np.concatenate(chains_post, axis=0).mean(axis=0)
        m = compute_metrics(theta_hat, data["theta_star"], z, chains_post,
                            wall_chain0, K_NDCG)
        metrics_by_method[method] = m
        print()
    return metrics_by_method


def aggregate_across_seeds(per_seed_metrics, methods, metric_keys):
    """seed 들의 평균/표준편차로 집계."""
    stats = {m: {} for m in methods}
    for method in methods:
        for key in metric_keys:
            vals = [per_seed_metrics[s][method][key]
                    for s in per_seed_metrics
                    if key in per_seed_metrics[s][method]]
            if len(vals) > 0:
                arr = np.array(vals, dtype=float)
                stats[method][key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std(ddof=0)),
                    "values": [float(v) for v in arr],
                }
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=[200, 1500, 10000],
                        required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    cfg = SCALE_CFG[args.n]
    T = cfg["T"]; batch = cfg["batch"]; K_NDCG = cfg["k_ndcg"]
    BURN_IN = int(T * BURN_IN_FRAC)
    n = args.n
    seeds = args.seeds

    print("=" * 76)
    print(f"Study 1C — sampler comparison  n={n}, seeds={seeds}")
    print(f"T={T}, BURN_IN={BURN_IN}, NUM_CHAINS={NUM_CHAINS}, batch={batch}, "
          f"k_NDCG={K_NDCG}")
    print(f"θ̂ = 3 chain pooled mean,  metric stats = mean±std over seeds")
    print("=" * 76)

    per_seed_metrics = {}
    for seed in seeds:
        print(f"\n=== seed {seed} ===")
        per_seed_metrics[seed] = run_one_seed(n, seed, T, BURN_IN,
                                              batch, K_NDCG)

    # 집계
    metric_keys = list(per_seed_metrics[seeds[0]][METHODS[0]].keys())
    stats = aggregate_across_seeds(per_seed_metrics, METHODS, metric_keys)

    # 출력
    print("\n" + "═" * 76)
    print(f"Aggregated  (mean ± std over {len(seeds)} seeds)")
    print("═" * 76)
    header_keys = ["mse_all", "spearman", f"ndcg_at_{K_NDCG}",
                   "R_hat_max", "ess_median", "cost_per_ess"]
    header = f"{'Method':<8} " + " ".join(
        [f"{k:>16}" for k in header_keys]
    )
    print(header)
    print("-" * len(header))
    for method in METHODS:
        line = f"{method:<8}"
        for key in header_keys:
            mu = stats[method][key]["mean"]
            sd = stats[method][key]["std"]
            line += f"  {mu:>7.3f}±{sd:.3f}"
        print(line)

    # save
    out = {
        "settings": {
            "n": n, "seeds": seeds, "T": T, "BURN_IN": BURN_IN,
            "NUM_CHAINS": NUM_CHAINS, "BATCH_SIZE": batch,
            "AWSGLD_SIGMA2_FLOOR": AWSGLD_SIGMA2_FLOOR,
            "SGHMC_LR_BASE": SGHMC_LR_BASE,
            "SGHMC_FRICTION": SGHMC_FRICTION,
            "SGHMC_TAU": SGHMC_TAU,
            "k_ndcg": K_NDCG,
            "theta_hat_definition": "3 chain pooled mean over post-burn",
        },
        "per_seed": {str(s): per_seed_metrics[s] for s in seeds},
        "aggregated": stats,
    }
    json_path = os.path.join(_THIS_DIR, f"results_n{n}_multiseed_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary -> {json_path}")

    png_path = os.path.join(_THIS_DIR, f"results_n{n}_multiseed.png")
    plot_summary_multiseed(stats, n, K_NDCG, len(seeds), png_path)


if __name__ == "__main__":
    main()
