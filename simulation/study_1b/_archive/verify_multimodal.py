"""
Study 1B — BSS posterior 가 실제로 multimodal 인지 확인.

방법
----
서로 다른 K 개 init 에서 BSS posterior energy 에 gradient descent 를 돌려
도달하는 stationary point 들이 서로 다른지 본다. 같은 점에 모두 모이면
unimodal, 2~3 개 군집으로 갈리면 multimodal.

설정
- σ² fixed (BSS 의 σ² floor 와 동일한 0.5)
- α 는 매 step alpha_find 로 grid-max (sampler 와 동일)
- 800 GD step, lr=0.02

Init 종류
- ini_N    : θ_i = μ_N (모두 N basin 에서 출발)
- ini_W    : θ_i = μ_W
- ini_S    : θ_i = μ_S
- ini_zero : θ_i = 0
- ini_truth: θ_i = θ_i^* (oracle, reference 용)
- ini_rand_k : 균등 [-3, 3] 무작위 init  k=0..2

각 final θ_hat 에서 group 별 mean(θ̂_g) 를 비교.

출력
- verify_multimodal_basins.png : group 별 final mean 의 init 별 점 그래프
- verify_multimodal_summary.json
"""
import json
import os
import sys

import numpy as np
from scipy.linalg import solve

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


_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


C_S, C_W, C_N = "#2F6DB2", "#D85A30", "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}

GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)
SIGMA2_FIXED = 0.5
N_STEPS = 800
LR_BASE = 0.02


def alpha_lk(theta, Y, alpha):
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    return float(np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp)))


def alpha_find(theta, Y, grid):
    return float(grid[int(np.argmax([alpha_lk(theta, Y, a) for a in grid]))])


def grad_energy(Y, alpha, theta, u_0, BtB, sigma2):
    """∂U/∂θ — full-batch (no noise, GD use)."""
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    dpi = pi * (1.0 - pi)
    seed_mask = Y == 1
    unl_mask = ~seed_mask
    grad_ll = np.zeros_like(theta)
    grad_ll[seed_mask] = 1.0 - pi[seed_mask]
    denom = np.clip(1.0 - (1.0 - alpha) * pi[unl_mask], 1e-10, None)
    grad_ll[unl_mask] = -(1.0 - alpha) * dpi[unl_mask] / denom
    grad_prior = -BtB @ (theta - u_0) / sigma2
    return -(grad_ll + grad_prior)  # ∂U = -∂ log p


def energy(Y, alpha, theta, u_0, B, sigma2):
    pi = sigmoid(theta)
    pi = np.clip(pi, 1e-10, 1 - 1e-10)
    temp = np.clip((1.0 - alpha) * pi, 1e-10, 1 - 1e-10)
    log_lik = np.sum(Y * np.log(temp) + (1.0 - Y) * np.log(1.0 - temp))
    diff = B @ (theta - u_0)
    log_prior = -0.5 / sigma2 * float(diff @ diff)
    return -(log_lik + log_prior)


def run_gd(data, ini, n_steps=N_STEPS, lr=LR_BASE):
    theta = ini.copy()
    Y, u_0, B = data["Y"], data["u_0"], data["B"]
    BtB = B.T @ B
    history = {"energy": [], "alpha": []}
    alpha = alpha_find(theta, Y, GRID)
    for t in range(n_steps):
        grad_U = grad_energy(Y, alpha, theta, u_0, BtB, SIGMA2_FIXED)
        eps = lr / ((t + 1) ** 0.4 + 5.0)
        theta = theta - eps * grad_U
        theta = np.clip(theta, -700, 700)
        alpha = alpha_find(theta, Y, GRID)
        if t % 50 == 0 or t == n_steps - 1:
            history["energy"].append(energy(Y, alpha, theta, u_0, B, SIGMA2_FIXED))
            history["alpha"].append(alpha)
    return theta, history


def main(seed=0):
    path = os.path.join(_THIS_DIR, f"data_seed{seed}.npz")
    d = np.load(path)
    n = int(d["n_total"])
    z = np.array([str(x) for x in d["z"]])
    data = {"Y": d["Y"], "u_0": d["u_0"], "B": d["B"]}
    theta_star = d["theta_star"]
    mu = PARAMS

    rng = np.random.default_rng(seed * 1000 + 17)
    inits = {
        "ini_N": np.full(n, mu["mu_N"]),
        "ini_W": np.full(n, mu["mu_W"]),
        "ini_S": np.full(n, mu["mu_S"]),
        "ini_zero": np.zeros(n),
        "ini_truth": theta_star.copy(),
        "ini_rand0": rng.uniform(-3, 3, size=n),
        "ini_rand1": rng.uniform(-3, 3, size=n),
        "ini_rand2": rng.uniform(-3, 3, size=n),
    }

    print(f"=== verify_multimodal on seed {seed} ===")
    print(f"n={n}, conflict={int(d['conflict_mask'].sum())}, "
          f"damping={float(d['damping'])}, p_in={float(d['p_in'])}, "
          f"p_out={float(d['p_out'])}")
    print(f"GD: σ²={SIGMA2_FIXED}, steps={N_STEPS}")
    print()

    results = {}
    for name, ini in inits.items():
        theta_final, hist = run_gd(data, ini)
        gms = {g: float(theta_final[z == g].mean()) for g in ("S", "W", "N")}
        mse = float(np.mean((theta_final - theta_star) ** 2))
        results[name] = {
            "init_mean": float(ini.mean()), "init_std": float(ini.std()),
            "final_group_mean": gms,
            "final_alpha": hist["alpha"][-1],
            "final_energy": hist["energy"][-1],
            "mse_truth": mse,
            "theta_final": theta_final,
        }
        print(
            f"[{name:11s}] init mean={ini.mean():+.2f}±{ini.std():.2f}  →  "
            f"final group mean (S/W/N)=({gms['S']:+.3f},{gms['W']:+.3f},"
            f"{gms['N']:+.3f})  "
            f"α={hist['alpha'][-1]:.3f}  E={hist['energy'][-1]:.1f}  "
            f"MSE(truth)={mse:.3f}"
        )

    # ── Multimodality 판정: final group mean (S, W, N) 의 init 간 spread
    print()
    spreads = {}
    for g in ("S", "W", "N"):
        vals = np.array([r["final_group_mean"][g] for r in results.values()])
        spreads[g] = {"min": float(vals.min()), "max": float(vals.max()),
                      "range": float(vals.max() - vals.min())}
        print(f"  group {g} final mean spread across inits: "
              f"[{vals.min():+.3f}, {vals.max():+.3f}]  range={spreads[g]['range']:.3f}")

    is_multimodal = any(spreads[g]["range"] > 0.3 for g in ("S", "W", "N"))
    print()
    print("=> POSTERIOR IS MULTIMODAL" if is_multimodal
          else "=> posterior appears UNIMODAL (all inits converge to same θ_hat)")

    # ── 시각화: init 별 group final mean 점 그래프
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    init_names = list(results.keys())
    x = np.arange(len(init_names))
    for ax, g in zip(axes, ("S", "W", "N")):
        vals = [results[k]["final_group_mean"][g] for k in init_names]
        ax.scatter(x, vals, color=GROUP_COLOR[g], s=70, zorder=3)
        ax.axhline(mu[f"mu_{g}"], color=GROUP_COLOR[g], ls="--",
                   label=fr"truth $\mu_{g}$={mu[f'mu_{g}']:+.2f}")
        for i, v in enumerate(vals):
            ax.annotate(f"{v:+.2f}", (x[i], v), fontsize=8,
                        ha="center", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(init_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel(fr"final mean($\hat\theta_{g}$) after GD")
        ax.set_title(f"Group {g}", fontweight="bold")
        ax.grid(True, alpha=0.2); ax.legend(fontsize=9)
    fig.suptitle(
        f"BSS posterior multimodality check (seed={seed}, n={n}, "
        f"d={float(d['damping'])}, conflict={int(d['conflict_mask'].sum())})\n"
        + ("→ MULTIMODAL" if is_multimodal else "→ unimodal"),
        fontweight="bold", fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_png = os.path.join(_THIS_DIR, "verify_multimodal_basins.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_png}")

    summary = {
        "seed": seed,
        "is_multimodal": bool(is_multimodal),
        "spreads": spreads,
        "results": {k: {kk: vv for kk, vv in r.items() if kk != "theta_final"}
                    for k, r in results.items()},
        "settings": {
            "n_steps": N_STEPS, "lr_base": LR_BASE,
            "sigma2_fixed": SIGMA2_FIXED,
            "damping": float(d["damping"]),
            "p_in": float(d["p_in"]), "p_out": float(d["p_out"]),
            "flip_rate_S_to_0": float(d["flip_rate_S_to_0"]),
            "flip_rate_N_to_1": float(d["flip_rate_N_to_1"]),
        },
    }
    json_path = os.path.join(_THIS_DIR, "verify_multimodal_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
