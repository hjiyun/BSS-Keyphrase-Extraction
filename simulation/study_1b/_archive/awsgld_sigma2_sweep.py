"""
Study 1B — AWSGLD sigma2_floor sweep.

σ²_floor 를 [0.5, 0.75, 1.0, 1.5, 2.0, 3.0] 로 바꿔가며 AWSGLD 의 recovery 지표를
빠르게 비교. Spearman 과 MSE_all 의 trade-off sweet spot 찾기.

데이터: data_seed0.npz  (acmh_vs_awsgld.py 와 동일)
체인  : 3 chain (bad init μ_N + dispersed)
출력  : awsgld_sigma2_sweep.json,  awsgld_sigma2_sweep.png

실행
----
python3 simulation/study_1b/awsgld_sigma2_sweep.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
for _p in (_THIS_DIR, CODE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from local_trap_landscape import PARAMS, sigmoid  # noqa: E402
from keyphrase_functions_awsgld import gibbs_mh as gibbs_mh_awsgld  # noqa: E402


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()


T = 5000
BURN_IN = 500
NUM_CHAINS = 3
BATCH_SIZE = 100
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)
SIGMA2_FLOOR_VALUES = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def alpha_lk(theta, Y, alpha):
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    return float(np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp)))


def alpha_find(theta, Y, grid):
    return float(grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))])


def make_inits(n, num_chains, rng):
    inits = [np.full(n, PARAMS["mu_N"], dtype=float)]
    targets = [PARAMS["mu_W"], PARAMS["mu_S"], 0.0]
    for i in range(num_chains - 1):
        center = targets[i % len(targets)]
        inits.append(center + rng.normal(0.0, 0.3, size=n))
    return inits[:num_chains]


def gelman_rubin(chains_post):
    M = len(chains_post)
    if M < 2:
        return np.nan, np.nan
    L = min(c.shape[0] for c in chains_post)
    arrs = np.stack([c[:L] for c in chains_post], axis=0)
    chain_means = arrs.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B_over_L = ((chain_means - grand_mean) ** 2).sum(axis=0) / (M - 1)
    W = arrs.var(axis=1, ddof=1).mean(axis=0)
    var_hat = (L - 1) / L * W + B_over_L
    R = np.sqrt(np.clip(var_hat / np.maximum(W, 1e-12), 0, None))
    return float(np.median(R)), float(np.max(R))


def run_one(data, sigma2_floor, seed=0):
    n = int(data["n_total"])
    Y = data["Y"]
    B = data["B"]
    u_0 = data["u_0"]
    graph = {"n": n, "A": data["A"], "D": np.diag(data["A"].sum(axis=1))}
    rng = np.random.default_rng(seed + 7777)
    inits = make_inits(n, NUM_CHAINS, rng)

    chains = []
    wall_total = 0.0
    for c_idx, ini in enumerate(inits):
        np.random.seed(seed * 1000 + c_idx)
        alpha0 = alpha_find(ini, Y, GRID)
        t0 = time.perf_counter()
        res = gibbs_mh_awsgld(
            Burn_in=BURN_IN, T=T, ini=ini.copy(), n=n, graph=graph,
            Y=Y, B=B, u_0=u_0,
            alpha_est=alpha0, grid=GRID,
            batch_size=BATCH_SIZE, sigma2_floor=sigma2_floor,
            verbose=False,
        )
        wall_total += time.perf_counter() - t0
        chains.append(res["theta_store"])

    theta_star = data["theta_star"]
    z = np.array([str(x) for x in data["z"]])
    theta_hat = chains[0][BURN_IN:].mean(axis=0)

    out = {
        "sigma2_floor": float(sigma2_floor),
        "mse_all": float(np.mean((theta_hat - theta_star) ** 2)),
        "spearman": float(spearmanr(theta_star, theta_hat).statistic),
        "wall_total": wall_total,
    }
    for g in ("S", "W", "N"):
        m = z == g
        out[f"mse_{g}"] = float(np.mean((theta_hat[m] - theta_star[m]) ** 2))
        out[f"mean_{g}"] = float(theta_hat[m].mean())
    rh_med, rh_max = gelman_rubin([c[BURN_IN:] for c in chains])
    out["rhat_median"] = rh_med
    out["rhat_max"] = rh_max
    return out


def main():
    data = np.load(os.path.join(_THIS_DIR, "data_seed0.npz"))
    print(f"Sweep σ²_floor over {SIGMA2_FLOOR_VALUES}  ({NUM_CHAINS} chains × T={T})")
    print()

    rows = []
    for s2 in SIGMA2_FLOOR_VALUES:
        row = run_one(data, s2)
        rows.append(row)
        print(
            f"σ²={s2:>4.2f}  MSE_all={row['mse_all']:.3f}  "
            f"Spear={row['spearman']:.3f}  "
            f"mean(S,W,N)=({row['mean_S']:+.2f},{row['mean_W']:+.2f},"
            f"{row['mean_N']:+.2f})  "
            f"MSE_g=({row['mse_S']:.2f},{row['mse_W']:.2f},{row['mse_N']:.2f})  "
            f"R̂(med/max)={row['rhat_median']:.3f}/{row['rhat_max']:.3f}  "
            f"wall={row['wall_total']:.1f}s"
        )

    out_json = os.path.join(_THIS_DIR, "awsgld_sigma2_sweep.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"settings": {"T": T, "BURN_IN": BURN_IN, "NUM_CHAINS": NUM_CHAINS,
                                "BATCH_SIZE": BATCH_SIZE},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_json}")

    # ── plot ──
    s2_vals = [r["sigma2_floor"] for r in rows]
    mu = PARAMS
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    ax.plot(s2_vals, [r["mse_all"] for r in rows], "o-", color="#9B59B6")
    ax.set_xlabel(r"σ² floor"); ax.set_ylabel("MSE_all")
    ax.set_title("MSE_all (낮을수록 좋음)", fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax = axes[0, 1]
    ax.plot(s2_vals, [r["spearman"] for r in rows], "o-", color="#1D9E75")
    ax.set_xlabel(r"σ² floor"); ax.set_ylabel("Spearman")
    ax.set_title("Spearman (높을수록 좋음)", fontweight="bold")
    ax.grid(True, alpha=0.2)

    ax = axes[1, 0]
    for g, color in [("S", "#2F6DB2"), ("W", "#D85A30"), ("N", "#6B6B6B")]:
        ax.plot(s2_vals, [r[f"mean_{g}"] for r in rows], "o-",
                color=color, label=fr"{g}  (truth $\mu_{g}$={mu[f'mu_{g}']:+.2f})")
        ax.axhline(mu[f"mu_{g}"], color=color, ls="--", alpha=0.4)
    ax.set_xlabel(r"σ² floor"); ax.set_ylabel(r"mean $\hat\theta_g$")
    ax.set_title(r"group mean($\hat\theta_g$) vs $\mu_g$", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    ax = axes[1, 1]
    ax.plot(s2_vals, [r["rhat_median"] for r in rows], "o-",
            color="#444", label="R̂ median")
    ax.plot(s2_vals, [r["rhat_max"] for r in rows], "o-",
            color="#B23A48", label="R̂ max")
    ax.axhline(1.1, color="black", ls="--", alpha=0.4)
    ax.axhline(1.2, color="gray", ls="--", alpha=0.4)
    ax.set_xlabel(r"σ² floor"); ax.set_ylabel("R-hat")
    ax.set_title("R-hat (1.1 이하 매우 좋음, 1.2 이하 양호)", fontweight="bold")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.2)

    fig.suptitle("AWSGLD  σ²_floor sweep on data_seed0",
                 fontweight="bold", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(_THIS_DIR, "awsgld_sigma2_sweep.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")


if __name__ == "__main__":
    main()
