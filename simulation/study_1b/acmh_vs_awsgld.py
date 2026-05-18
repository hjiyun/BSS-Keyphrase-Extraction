"""
Study 1B — acMH vs AWSGLD 비교 (recovery + trap escape).

배경
----
- 데이터: data_generator.py 로 만든 data_seed{S}.npz 중 하나 (기본 seed=0).
- Target: local_trap_landscape.py 의 3-mode mixture posterior
  (mu_N/mu_W/mu_S = PARAMS 에서 import).
- N basin 이 가장 mass 가 큼 (rho_N=0.6), 그 위로 W 와 S basin.
- 평가 목적은 두 가지:
  A. Recovery — theta* 를 얼마나 복원하는가 (특히 group 별).
  B. Trap escape — bad init 에서 S/W mode 로 빠져나오는 능력.

샘플러
------
- acMH      : keyphrase_functions.gibbs_mh (Metropolis-Hastings within Gibbs,
              원본 BSS). proposal cov ∝ (B^T B)^{-1} · σ² · 4/n.
- AWSGLD    : keyphrase_functions_awsgld.gibbs_mh (preconditioned SGLD +
              adaptive weighting over energy partitions + σ² Gibbs floor).

샘플링 설정 (스크립트 상단 상수)
-------------------------------
- T              = 5000   (chain 길이)
- BURN_IN        = 500    (recovery 평균낼 때 cut)
- NUM_CHAINS     = 3      (Gelman-Rubin R-hat 용)
- BATCH_SIZE     = 100    (n=400 의 25% — 두 샘플러 모두 동일하게 적용)
- AWSGLD partition (M_REGIONS) = 1000  (keyphrase_functions_awsgld.py 의
  소스 상수. AWSGLD 가 energy 공간을 1000 개 구간으로 나누어 adaptive weight
  를 유지함. 이 스크립트에서 변경하지 않음.)

지표
----
A. Recovery
   - MSE(theta_hat, theta*) 전체 평균
   - Group 별 MSE (S/W/N)
   - Spearman corr(theta_hat, theta*)
   - Group mean(theta_hat) vs mu_g  (3-mode 분리 회복 여부)

B. Trap escape (bad init: 모든 theta_i^(0) = mu_N from PARAMS)
   - Trace plot: 각 group 대표 노드 몇 개의 theta_i^(t) 시간 경로
   - Escape time: theta_i 가 자기 mode 의 ±0.5 안에 처음 들어오는 step
   - Mode visit count: theta_i^(t) 가 어느 basin 안에 머문 step 수
   - R-hat: NUM_CHAINS dispersed init 의 between/within chain 분산 비율

출력
----
- ava_results.npz           : 모든 chain 의 theta_store, metric 값
- ava_recovery.png          : group 별 회복 (parity + bar)
- ava_trace_<method>.png    : group 대표 노드 trace
- ava_mode_visit.png        : mode visit count bar
- ava_metric_summary.json   : 핵심 지표 요약

실행
----
python3 simulation/study_1b/acmh_vs_awsgld.py            # seed=0
python3 simulation/study_1b/acmh_vs_awsgld.py 2          # seed=2
"""
import json
import os
import sys
import time

import numpy as np
from scipy.linalg import solve
from scipy.stats import spearmanr

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ── 한글 폰트 ──
_FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT_PATH):
    fm.fontManager.addfont(_FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT_PATH).get_name()
plt.rcParams["axes.unicode_minus"] = False


# 경로 셋업
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(_THIS_DIR))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
ORIG_DIR = os.path.join(CODE_DIR, "original")
for _p in (_THIS_DIR, CODE_DIR, ORIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from local_trap_landscape import PARAMS, sigmoid  # noqa: E402
from keyphrase_functions import gibbs_mh as gibbs_mh_acmh  # noqa: E402
from keyphrase_functions_awsgld import gibbs_mh as gibbs_mh_awsgld  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 샘플링 설정
# ──────────────────────────────────────────────────────────────────────────
T = 5000
BURN_IN = 500
NUM_CHAINS = 3
BATCH_SIZE = 100
AWSGLD_SIGMA2_FLOOR = 1.0  # AWSGLD σ² 하한 (default 0.5).
                           # awsgld_sigma2_sweep.py 의 sweep 결과 rank-priority
                           # sweet spot: Spearman 0.697 (cycSGLD 0.692 를 +0.005
                           # 차이로 이김) + R̂ 1.15 + ESS 19+ + cost-per-ESS 최저.
                           # MSE_all 1.382 로 cycSGLD 의 1.272 에 2위.
GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)
DAMPING = 0.85  # data_generator 와 동일

# CLAUDE.md 색상 규칙
C_S = "#2F6DB2"
C_W = "#D85A30"
C_N = "#6B6B6B"
GROUP_COLOR = {"S": C_S, "W": C_W, "N": C_N}
METHOD_COLOR = {"acMH": "#E1B12C", "AWSGLD": "#9B59B6"}


# ──────────────────────────────────────────────────────────────────────────
# 데이터 로딩
# ──────────────────────────────────────────────────────────────────────────
def load_data(seed):
    path = os.path.join(_THIS_DIR, f"data_seed{seed}.npz")
    d = np.load(path)
    return {
        "theta_star": d["theta_star"],
        "Y": d["Y"],
        "z": np.array([str(x) for x in d["z"]]),
        "A": d["A"],
        "B": d["B"],
        "u_0": d["u_0"],
        "graph": {"n": int(d["n_total"]), "A": d["A"],
                  "D": np.diag(d["A"].sum(axis=1))},
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
# 초기값 전략
# ──────────────────────────────────────────────────────────────────────────
def make_inits(n, num_chains, rng):
    """
    Chain 0: 모든 노드 theta=mu_N (bad init, trap escape 시험용 anchor).
    Chain 1..k: dispersed — 각각 mu_W, mu_S, 그리고 N(0, 1).
    Recovery 평가는 chain 0 의 burn-in 뒤를 기본으로 사용 (가장 어려운 출발).
    R-hat 은 num_chains 모두 사용.
    """
    inits = [np.full(n, PARAMS["mu_N"], dtype=float)]
    targets = [PARAMS["mu_W"], PARAMS["mu_S"], 0.0]
    for i in range(num_chains - 1):
        center = targets[i % len(targets)]
        inits.append(center + rng.normal(0.0, 0.3, size=n))
    return inits[:num_chains]


# ──────────────────────────────────────────────────────────────────────────
# 샘플러 wrapper (동일 입력 → theta_store 반환)
# ──────────────────────────────────────────────────────────────────────────
def run_acmh(data, ini, verbose=False):
    t0 = time.perf_counter()
    alpha0 = alpha_find(ini, data["Y"], GRID)
    res = gibbs_mh_acmh(
        Burn_in=BURN_IN, T=T, ini=ini.copy(), n=data["n"], graph=data["graph"],
        Y=data["Y"], B=data["B"], u_0=data["u_0"],
        alpha_est=alpha0, grid=GRID, verbose=verbose,
    )
    return {
        "theta_store": res["theta_store"],
        "alpha_store": np.array([float(res["alpha_mn"])] * T),
        "wall_time": time.perf_counter() - t0,
    }


def run_awsgld(data, ini, verbose=False):
    t0 = time.perf_counter()
    alpha0 = alpha_find(ini, data["Y"], GRID)
    res = gibbs_mh_awsgld(
        Burn_in=BURN_IN, T=T, ini=ini.copy(), n=data["n"], graph=data["graph"],
        Y=data["Y"], B=data["B"], u_0=data["u_0"],
        alpha_est=alpha0, grid=GRID,
        batch_size=BATCH_SIZE, sigma2_floor=AWSGLD_SIGMA2_FLOOR,
        verbose=verbose,
    )
    return {
        "theta_store": res["theta_store"],
        "alpha_store": np.array([float(res["alpha_mn"])] * T),
        "wall_time": time.perf_counter() - t0,
    }


METHOD_RUNNERS = {"acMH": run_acmh, "AWSGLD": run_awsgld}


# ──────────────────────────────────────────────────────────────────────────
# 지표 — A. Recovery
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


# ──────────────────────────────────────────────────────────────────────────
# 지표 — B. Trap escape
# ──────────────────────────────────────────────────────────────────────────
def escape_time_per_node(theta_store, z, mu_map, window=0.5):
    """각 노드가 자기 group 의 mu_g ±window 안에 처음 들어온 step. 못 들어오면 nan."""
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
    """전체 (burn-in 후) chain 에서 각 mode 영역에 머문 step 수 (총합 over nodes)."""
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
    """
    chains_theta_post: list of (T_post, n) arrays.
    노드별 R-hat 후 median/quantile 반환.
    """
    M = len(chains_theta_post)
    if M < 2:
        return {"R_hat_median": np.nan, "R_hat_max": np.nan, "R_hat_q90": np.nan}
    L = min(c.shape[0] for c in chains_theta_post)
    arrs = np.stack([c[:L] for c in chains_theta_post], axis=0)  # (M, L, n)
    chain_means = arrs.mean(axis=1)                              # (M, n)
    grand_mean = chain_means.mean(axis=0)                        # (n,)
    B_over_L = ((chain_means - grand_mean) ** 2).sum(axis=0) / (M - 1)
    W = arrs.var(axis=1, ddof=1).mean(axis=0)                    # (n,)
    var_hat = (L - 1) / L * W + B_over_L
    R = np.sqrt(np.clip(var_hat / np.maximum(W, 1e-12), 0, None))
    return {
        "R_hat_median": float(np.median(R)),
        "R_hat_max": float(np.max(R)),
        "R_hat_q90": float(np.quantile(R, 0.90)),
        "R_hat_per_node": R,
    }


# ──────────────────────────────────────────────────────────────────────────
# 시각화
# ──────────────────────────────────────────────────────────────────────────
def plot_recovery(data, theta_hat_by_method, out_path):
    z = data["z"]
    theta_star = data["theta_star"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (1, 1) Parity plots (theta_hat vs theta_star) 한 패널씩
    for col, method in enumerate(theta_hat_by_method.keys()):
        ax = axes[0, col]
        theta_hat = theta_hat_by_method[method]
        for g in ("N", "W", "S"):
            m = z == g
            ax.scatter(theta_star[m], theta_hat[m], color=GROUP_COLOR[g],
                       label=f"{g} (n={int(m.sum())})", s=16, alpha=0.7)
        lo = min(theta_star.min(), theta_hat.min()) - 0.5
        hi = max(theta_star.max(), theta_hat.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, lw=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel(r"$\theta^*$"); ax.set_ylabel(r"$\hat{\theta}$")
        ax.set_title(f"{method} — parity (chain0 from bad init)",
                     fontweight="bold", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.15)

    # (1, 0) Group MSE bar
    ax = axes[1, 0]
    methods = list(theta_hat_by_method.keys())
    groups = ("S", "W", "N")
    x = np.arange(len(groups))
    width = 0.38
    for i, method in enumerate(methods):
        mses = [np.mean((theta_hat_by_method[method][z == g] - theta_star[z == g]) ** 2)
                for g in groups]
        ax.bar(x + (i - 0.5) * width, mses, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel("Group MSE")
    ax.set_title("Group 별 MSE (낮을수록 좋음)", fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y"); ax.legend(fontsize=9)

    # (1, 1) Group mean(theta_hat) vs mu_g
    ax = axes[1, 1]
    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}
    for i, method in enumerate(methods):
        means = [theta_hat_by_method[method][z == g].mean() for g in groups]
        ax.bar(x + (i - 0.5) * width, means, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    # ground truth 선
    for j, g in enumerate(groups):
        ax.hlines(mu_map[g], j - 0.5, j + 0.5, colors=GROUP_COLOR[g],
                  linestyles="--", linewidth=1.8,
                  label=fr"$\mu_{g}$={mu_map[g]}" if i == len(methods) - 1 else None)
    ax.set_xticks(x); ax.set_xticklabels(groups)
    ax.set_ylabel(r"mean $\hat{\theta}$")
    ax.set_title(r"Group mean($\hat{\theta}$) vs $\mu_g$  (3-mode 분리 회복)",
                 fontweight="bold", fontsize=11)
    ax.grid(True, alpha=0.15, axis="y"); ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"Study 1B — acMH vs AWSGLD recovery (seed={data['seed']}, "
        f"n={data['n']}, T={T}, burn={BURN_IN}, bad init=$\\mu_N$)",
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
                    lw=0.9,
                    label=(f"{g} (node {i})" if j == 0 else None))
        ax.axhline(mu_map[g], color=GROUP_COLOR[g], ls="--", lw=1.2, alpha=0.8)
    ax.axvline(BURN_IN, color="black", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("MCMC step"); ax.set_ylabel(r"$\theta_i^{(t)}$")
    ax.set_title(
        f"{method} — trace from bad init ($\\theta^{{(0)}}=\\mu_N=-1.0$). "
        f"점선 = 각 group $\\mu_g$, 검은 점선 = burn-in 끝",
        fontweight="bold", fontsize=11,
    )
    ax.grid(True, alpha=0.15); ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


def plot_mode_visit(visit_by_method, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    methods = list(visit_by_method.keys())
    cats = ("S", "W", "N", "other")
    x = np.arange(len(cats))
    width = 0.38
    for i, method in enumerate(methods):
        vals = [visit_by_method[method][c] for c in cats]
        total = sum(vals)
        frac = [v / total for v in vals] if total > 0 else vals
        ax.bar(x + (i - 0.5) * width, frac, width,
               color=METHOD_COLOR[method], alpha=0.85, label=method)
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylabel("fraction of post-burn samples in basin (across all nodes)")
    ax.set_title("Mode visit count (bad init chain) — basin coverage",
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
    print(f"Study 1B — acMH vs AWSGLD on data_seed{seed}.npz")
    print(f"n={n}, T={T}, burn={BURN_IN}, num_chains={NUM_CHAINS}, "
          f"batch_size={BATCH_SIZE}, AWSGLD M_REGIONS=1000")
    print(f"Groups: S={int((z=='S').sum())}, W={int((z=='W').sum())}, "
          f"N={int((z=='N').sum())}")
    print("=" * 72)

    mu_map = {"S": PARAMS["mu_S"], "W": PARAMS["mu_W"], "N": PARAMS["mu_N"]}
    rng = np.random.default_rng(seed + 7777)
    inits = make_inits(n, NUM_CHAINS, rng)

    all_chains = {m: [] for m in METHOD_RUNNERS}
    wall_times = {m: [] for m in METHOD_RUNNERS}

    for method, runner in METHOD_RUNNERS.items():
        print(f"\n── {method} ──")
        for c_idx, ini in enumerate(inits):
            np.random.seed(seed * 1000 + c_idx)  # 내부 random 도 시드
            print(f"  chain {c_idx}: init mean={ini.mean():+.3f} std={ini.std():.3f}",
                  flush=True)
            res = runner(data, ini, verbose=False)
            all_chains[method].append(res["theta_store"])
            wall_times[method].append(res["wall_time"])
            print(f"    -> wall {res['wall_time']:.1f}s")

    # ── Recovery metrics (chain 0: bad init) ──
    print("\n" + "─" * 72)
    print("A. Recovery metrics  (chain 0, theta_hat = mean over post-burn-in)")
    print("─" * 72)
    theta_hat_by_method = {}
    recovery_by_method = {}
    for method in METHOD_RUNNERS:
        ts = all_chains[method][0]
        theta_hat = ts[BURN_IN:].mean(axis=0)
        theta_hat_by_method[method] = theta_hat
        rec = recovery_metrics(theta_hat, data["theta_star"], z)
        recovery_by_method[method] = rec
        print(f"[{method}]  MSE_all={rec['mse_all']:.3f}  "
              f"Spearman={rec['spearman']:.3f}")
        for g in ("S", "W", "N"):
            print(f"   {g}: MSE={rec[f'mse_{g}']:.3f}  "
                  f"mean(theta_hat)={rec[f'mean_theta_hat_{g}']:+.3f}  "
                  f"(target mu_{g}={mu_map[g]:+.2f}, "
                  f"true mean={rec[f'mean_theta_star_{g}']:+.3f})")

    # ── Trap escape ──
    print("\n" + "─" * 72)
    print("B. Trap escape  (chain 0 from bad init theta^(0)=mu_N)")
    print("─" * 72)
    escape_by_method = {}
    mode_visit_by_method = {}
    for method in METHOD_RUNNERS:
        ts = all_chains[method][0]
        et = escape_time_per_node(ts, z, mu_map, window=0.5)
        escape_by_method[method] = et
        mv = mode_visit_counts(ts, mu_map, window=0.5)
        mode_visit_by_method[method] = mv
        print(f"[{method}]  Escape time (median ± IQR, ‘nan’ = never escaped):")
        for g in ("S", "W", "N"):
            sub = et[z == g]
            n_ok = int(np.sum(~np.isnan(sub)))
            n_all = int(np.sum(z == g))
            if n_ok > 0:
                med = float(np.nanmedian(sub))
                q25 = float(np.nanquantile(sub, 0.25))
                q75 = float(np.nanquantile(sub, 0.75))
                print(f"   {g}: {n_ok}/{n_all} escaped  "
                      f"median={med:.0f} step  IQR=[{q25:.0f}, {q75:.0f}]")
            else:
                print(f"   {g}: 0/{n_all} escaped within {T} steps")
        print(f"  Mode visit fractions (post-burn, across all nodes):")
        total = sum(mv.values())
        for cat in ("S", "W", "N", "other"):
            print(f"   {cat}: {mv[cat]/total*100:.1f}%")

    # ── R-hat ──
    print("\n" + "─" * 72)
    print("C. R-hat (Gelman-Rubin)  — between/within chain variance ratio")
    print("─" * 72)
    rhat_by_method = {}
    for method in METHOD_RUNNERS:
        post_chains = [ts[BURN_IN:] for ts in all_chains[method]]
        rh = gelman_rubin(post_chains)
        rhat_by_method[method] = rh
        print(f"[{method}]  R_hat median={rh['R_hat_median']:.3f}  "
              f"q90={rh['R_hat_q90']:.3f}  max={rh['R_hat_max']:.3f}")

    # ── 시각화 ──
    print("\n" + "─" * 72)
    print("Saving plots ...")
    print("─" * 72)
    plot_recovery(data, theta_hat_by_method,
                  os.path.join(_THIS_DIR, "ava_recovery.png"))
    for method in METHOD_RUNNERS:
        plot_trace(all_chains[method][0], z, method,
                   os.path.join(_THIS_DIR, f"ava_trace_{method}.png"),
                   k_per_group=3, seed=seed)
    plot_mode_visit(mode_visit_by_method,
                    os.path.join(_THIS_DIR, "ava_mode_visit.png"))

    # ── 결과 저장 ──
    npz_path = os.path.join(_THIS_DIR, "ava_results.npz")
    npz_kwargs = {
        "theta_star": data["theta_star"], "Y": data["Y"], "z": data["z"],
        "seed": np.int64(seed), "T": np.int64(T), "BURN_IN": np.int64(BURN_IN),
        "NUM_CHAINS": np.int64(NUM_CHAINS), "BATCH_SIZE": np.int64(BATCH_SIZE),
    }
    for method in METHOD_RUNNERS:
        for c_idx, ts in enumerate(all_chains[method]):
            npz_kwargs[f"{method}_chain{c_idx}_theta_store"] = ts
        npz_kwargs[f"{method}_theta_hat"] = theta_hat_by_method[method]
        npz_kwargs[f"{method}_escape_time"] = escape_by_method[method]
        npz_kwargs[f"{method}_Rhat_per_node"] = rhat_by_method[method]["R_hat_per_node"]
    np.savez(npz_path, **npz_kwargs)
    print(f"Saved -> {npz_path}")

    # ── JSON summary ──
    summary = {
        "settings": {
            "seed": seed, "n": n, "T": T, "BURN_IN": BURN_IN,
            "NUM_CHAINS": NUM_CHAINS, "BATCH_SIZE": BATCH_SIZE,
            "AWSGLD_M_REGIONS": 1000,
            "bad_init": "theta^(0) = mu_N = -1.0 (모든 노드)",
            "escape_window": 0.5,
        },
        "groups": {"S": int((z == "S").sum()),
                   "W": int((z == "W").sum()),
                   "N": int((z == "N").sum())},
        "mu_map": mu_map,
        "recovery": recovery_by_method,
        "mode_visit": mode_visit_by_method,
        "wall_time_sec": {m: wall_times[m] for m in METHOD_RUNNERS},
        "R_hat": {m: {k: v for k, v in rhat_by_method[m].items()
                      if k != "R_hat_per_node"}
                  for m in METHOD_RUNNERS},
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
            for method in METHOD_RUNNERS
        },
    }
    json_path = os.path.join(_THIS_DIR, "ava_metric_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved -> {json_path}")


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(seed)
