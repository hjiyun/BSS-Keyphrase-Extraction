"""cyc_awsgld.png 스타일 strip plot — 단, y='최저 U'가 아니라 'cut-off 에너지 도달 반복수'.

각 run 은 에너지가 cut-off 선(U≤CUT)에 도달하면 '멈춘' 것으로 보고, 그때까지의 반복수를 기록.
seed 별 점 + 평균 막대(= cyc_awsgld.png 형식). AWSGLD vs cycSGLD.
왼쪽: 도달 반복수(둘 다 도달 → 비슷/누가 빠른가) | 오른쪽: ESS(같은 형식).

출력: strip_cutoff_ess.png / .csv
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
DATA_SEEDS = [0, 1, 2, 3, 4]
RNG_PER = [0, 1]                 # 데이터 seed 당 RNG 2개 → 총 10점 (cyc_awsgld.png 와 동일 규모)
CUT = 320                        # 공통 cut-off 에너지선 (수렴 영역 ~312-320)
COL = {"AWSGLD": "#2456A6", "cycSGLD": "#8e44ad"}
METHODS = ["AWSGLD", "cycSGLD"]


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"]


def run(method, ini, Y, B, u_0, a0, n, BtB, P, Lc, rng_seed):
    np.random.seed(rng_seed); theta = ini.copy(); ths = np.zeros((T, n)); alpha = a0
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
    idx = np.where(U_tr <= CUT)[0]
    reach = int(idx[0]) if len(idx) else T          # 미도달 시 T
    ess = float(np.nanmedian(ess_per_node(ths[BURN:])))
    return reach, ess


def main():
    reach = {m: [] for m in METHODS}; ess = {m: [] for m in METHODS}
    t0 = time.time()
    print(f"strip: cut-off U≤{CUT} 도달 반복수 + ESS | {len(DATA_SEEDS)*len(RNG_PER)}점/method\n", flush=True)
    for s in DATA_SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
        ini = np.full(n, float(E.MU_N))
        for r in RNG_PER:
            for m in METHODS:
                rc, es = run(m, ini, Y, B, u_0, a0, n, BtB, P, Lc, rng_seed=1000 * s + r)
                reach[m].append(rc); ess[m].append(es)
        print(f"  seed {s} done ({int(time.time()-t0)}s)", flush=True)

    for m in METHODS:
        print(f"  {m:>8}: reach mean={np.mean(reach[m]):.0f} (min {min(reach[m])}, max {max(reach[m])})  "
              f"ESS mean={np.mean(ess[m]):.1f}")

    with open(os.path.join(HERE, "strip_cutoff_ess.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "point", "reach_iters", "ess"])
        for m in METHODS:
            for i in range(len(reach[m])):
                w.writerow([m, i, reach[m][i], round(ess[m][i], 2)])

    # ── cyc_awsgld.png 스타일 strip: 점 + 평균 막대 + 값 라벨 ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    rng = np.random.RandomState(0)
    for ax, dic, ylab, ttl, fmt in [
            (axes[0], reach, f"iterations to reach cut-off (U≤{CUT})",
             f"(a) time to reach cut-off energy", "{:.0f}"),
            (axes[1], ess, "ESS (median over nodes)", "(b) ESS", "{:.1f}")]:
        for i, m in enumerate(METHODS):
            v = np.array(dic[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
            ax.scatter(xs, v, s=46, color=COL[m], alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
            mean = v.mean()
            ax.plot([i - 0.28, i + 0.28], [mean, mean], color=COL[m], lw=3, zorder=4)
            ax.text(i, v.max() + (v.max() - v.min() + 1) * 0.04, fmt.format(mean),
                    ha="center", va="bottom", fontsize=11, color=COL[m], fontweight="bold")
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=11)
        for lbl, m in zip(ax.get_xticklabels(), METHODS):
            lbl.set_color(COL[m]); lbl.set_fontweight("bold")
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"BSS posterior — stop at cut-off U≤{CUT}: reaching it is comparable, ESS favors AWSGLD",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strip_cutoff_ess.png"), dpi=140, bbox_inches="tight")
    print(f"\n저장: strip_cutoff_ess.png, strip_cutoff_ess.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
