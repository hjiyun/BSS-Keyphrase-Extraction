"""BSS: cut-off 를 사람이 고르지 않고 '수렴 지표(running R̂ + 에너지 plateau)'가 정하게.

절차
----
- 3 chain (과분산 시작점 μ_N/μ_W/μ_S) 으로 AWSGLD, cycSGLD 실행.
- running R̂: 반복수를 늘려가며 R̂max(노드별 최댓값) 계산 → 처음 <1.1 되는 반복수 = '수렴 시점'.
- 에너지 plateau: pooled 에너지 running-median 기울기가 평탄해지는 반복수 자동 검출.
- cut-off 에너지 = 수렴 시점 이후 정상상태 에너지 median. (사람이 320 고르지 않음)
- ESS: 수렴 후 표본 효율.

핵심 정직한 관찰: R̂ gate 로 판정하면 AWSGLD 만 <1.1 로 수렴, cycSGLD 는 계속 >1.1(collapse).
출력: convergence_cutoff.png / .csv
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5000; BURN = 500; BATCH = 100; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
CYC = 10
SEEDS = [0, 1, 2]
INITS = [("μ_N", -0.8), ("μ_W", 1.0), ("μ_S", 2.5)]   # 과분산 3 시작점
COL = {"AWSGLD": "#2456A6", "cycSGLD": "#8e44ad"}
RHAT_THR = 1.2   # 널리 쓰이는 수렴 기준(1.1 은 더 엄격). AWSGLD R̂≈1.14 → 1.2 통과, 1.1 아슬


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"]


def run(method, ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 20)); es = []; emin = du = None; J = M_REG - 1
    use_aw = method == "AWSGLD"
    for t in range(T):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        if method == "cycSGLD":
            cl = max(1, T // CYC); beta = (t % cl) / cl
            eps = 0.02 / 2.0 * (np.cos(np.pi * min(beta, 0.8)) + 1); tau_k = TAU if beta >= 0.8 else TAU / 1e4
        else:
            eps = 0.3 / ((t + 1) ** 0.6 + 10); tau_k = TAU
        U = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2)
        bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, alpha, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        gm = 1.0
        if use_aw:
            if t < warm:
                es.append(U)
                if t == warm - 1:
                    lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                    emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
            else:
                J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
                gm = float(np.clip(1 + (ZETA * tau_k / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        if use_aw:
            theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * tau_k * eps) * (Lc @ np.random.randn(n))
        else:
            theta = theta - eps * gU + np.sqrt(2 * tau_k * eps) * np.random.randn(n)
        theta = np.clip(theta, -700, 700)
        if use_aw and t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; alpha = E.alpha_find(theta, Y, E.GRID)
    U_tr = E.energy_trace_common(ths, Y, B, u_0, a0)
    return ths, U_tr


def rhat_max(chains, lo, hi):
    """chains: list of (T,n). [lo:hi] 구간으로 노드별 R̂ 계산, 최댓값 반환."""
    arrs = np.stack([c[lo:hi] for c in chains], 0)          # (M, L, n)
    M, L, n = arrs.shape
    cm = arrs.mean(1); gm = cm.mean(0)
    B_over_L = ((cm - gm) ** 2).sum(0) / (M - 1)
    W = arrs.var(1, ddof=1).mean(0)
    var_hat = (L - 1) / L * W + B_over_L
    R = np.sqrt(np.clip(var_hat / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R))


def running_rhat(chains, checkpoints):
    """각 checkpoint tc 에서 고정 burn 이후 [BURN:tc] 로 R̂max (프로젝트 표준 정의)."""
    out = []
    for tc in checkpoints:
        lo = BURN
        out.append(rhat_max(chains, lo, tc) if tc - lo >= 3 else np.nan)
    return np.array(out)


def plateau_start(U, win=200, tol=3.0):
    """running-median 기울기 평탄 지점 자동 검출."""
    for t in range(win, len(U) - win):
        if abs(np.median(U[t - win:t]) - np.median(U[t:t + win])) < tol:
            return t
    return None


def main():
    METHODS = ["AWSGLD", "cycSGLD"]
    checkpoints = np.unique(np.linspace(BURN+100, T, 25).astype(int))
    agg = {m: {"conv_iter": [], "cutoff_E": [], "ess": [], "rhat_final": [], "plateau": []} for m in METHODS}
    demo = {}   # seed0 트레이스/러닝R̂ 저장(그림)
    t0 = time.time()
    print(f"수렴지표 기반 cut-off | AWSGLD vs cycSGLD | seeds={SEEDS} | 3 chains {[' '.join([i[0] for i in INITS])]}\n", flush=True)
    for s in SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
        for m in METHODS:
            chains = []; Utrs = []
            for ci, (nm, val) in enumerate(INITS):
                ths, U_tr = run(m, np.full(n, float(val)), Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * s + ci)
                chains.append(ths); Utrs.append(U_tr)
            rr = running_rhat(chains, checkpoints)
            below = np.where(rr < RHAT_THR)[0]
            conv_iter = int(checkpoints[below[0]]) if len(below) else None
            pooledU = np.concatenate([U[BURN:] for U in Utrs])
            plat = plateau_start(Utrs[0])
            # cut-off 에너지: 수렴 시점 이후(없으면 plateau 이후) 정상상태 median
            ref = conv_iter if conv_iter else (plat if plat else BURN)
            cutoffE = float(np.median(np.concatenate([U[ref:] for U in Utrs])))
            ess = float(np.nanmedian([np.nanmedian(ess_per_node(c[BURN:])) for c in chains]))
            agg[m]["conv_iter"].append(conv_iter); agg[m]["cutoff_E"].append(cutoffE)
            agg[m]["ess"].append(ess); agg[m]["rhat_final"].append(float(rr[-1])); agg[m]["plateau"].append(plat)
            if s == 0: demo[m] = (Utrs, rr, conv_iter, plat, cutoffE)
            ci_str = f"{conv_iter}" if conv_iter else "미수렴"
            print(f"  seed {s} {m:>8}: R̂수렴={ci_str:>6}  cutoffE={cutoffE:>4.0f}  R̂final={rr[-1]:.2f}  ESS={ess:.1f}", flush=True)

    print("\n=== 요약 (평균) ===")
    for m in METHODS:
        ci = [x for x in agg[m]["conv_iter"] if x is not None]
        ci_str = f"{np.mean(ci):.0f} ({len(ci)}/{len(SEEDS)} 수렴)" if ci else f"미수렴 (0/{len(SEEDS)})"
        print(f"  {m:>8}: R̂수렴시점={ci_str:>16}  cutoffE={np.mean(agg[m]['cutoff_E']):.0f}"
              f"  R̂final={np.mean(agg[m]['rhat_final']):.2f}  ESS={np.mean(agg[m]['ess']):.1f}")
    print(f"  ({int(time.time()-t0)}s)")

    with open(os.path.join(HERE, "convergence_cutoff.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "rhat_conv_iter", "cutoff_energy", "rhat_final", "ess", "plateau_iter"])
        for m in METHODS:
            for i, s in enumerate(SEEDS):
                w.writerow([m, s, agg[m]["conv_iter"][i], round(agg[m]["cutoff_E"][i], 1),
                            round(agg[m]["rhat_final"][i], 3), round(agg[m]["ess"][i], 2), agg[m]["plateau"][i]])

    # ── 그림: 2행(AWSGLD/cycSGLD) × 2열(에너지 trace+plateau/cutoff | running R̂) + ESS 텍스트 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, m in enumerate(METHODS):
        Utrs, rr, conv_iter, plat, cutoffE = demo[m]
        ax = axes[row, 0]
        for U in Utrs: ax.plot(U, color=COL[m], lw=0.5, alpha=0.55)
        ax.axhline(cutoffE, color="k", ls="--", lw=1.3, alpha=0.8, label=f"cut-off E={cutoffE:.0f}")
        if plat: ax.axvline(plat, color="#2E9E5B", ls=":", lw=1.4, alpha=0.8, label=f"plateau@{plat}")
        if conv_iter: ax.axvline(conv_iter, color="#C0392B", ls="-", lw=1.4, alpha=0.8, label=f"R̂<1.2@{conv_iter}")
        ax.set_ylabel(f"{m}\nU(theta)"); ax.set_title(f"({'a' if row==0 else 'c'}) {m}: energy traces (3 chains)", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.15)
        if row == 1: ax.set_xlabel("iteration")
        ax = axes[row, 1]
        ax.plot(np.unique(np.linspace(BURN+100, T, 25).astype(int)), rr, "o-", color=COL[m], lw=1.5, ms=4)
        ax.axhline(RHAT_THR, color="red", ls="--", lw=1, alpha=0.7, label="R̂=1.2")
        ax.set_ylabel("R̂ max (running)"); ax.set_title(f"({'b' if row==0 else 'd'}) {m}: running R̂  (final {rr[-1]:.2f})", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(alpha=0.2)
        if row == 1: ax.set_xlabel("iteration up to")
    essA = np.mean(agg["AWSGLD"]["ess"]); essC = np.mean(agg["cycSGLD"]["ess"])
    fig.suptitle(f"Data-driven cut-off via convergence: only AWSGLD reaches R̂<1.1 | ESS  AWSGLD {essA:.0f} vs cycSGLD {essC:.0f}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "convergence_cutoff.png"), dpi=140, bbox_inches="tight")
    print("저장: convergence_cutoff.png, convergence_cutoff.csv")


if __name__ == "__main__":
    main()
