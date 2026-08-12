"""BSS: 공통 cut-off 에너지선 '도달 시간(first-passage)' + ESS 로 AWSGLD vs cycSGLD.

아이디어 (사용자 제안)
--------------------
Rastrigin 그림처럼 '가장 낮은 U'가 아니라 '목표 cut-off 에너지에 도달할 때까지'를 잰다.
bad init(μ_N)에서 출발해 에너지가 높은 상태 → 수렴 영역으로 내려오며 cut-off 선을 통과.
공통 cut-off 를 둘 다 통과하면 '도달'은 비슷 → 차이는 ESS(효율)에서 난다.

cut-off 는 한 값에 의존하지 않도록 여러 레벨(sweep)에서 도달 반복수를 함께 본다.
출력: reach_time_ess.png / .csv
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
SEEDS = [0, 1, 2, 3, 4]
COL = {"AWSGLD": "#2456A6", "cycSGLD": "#8e44ad"}
CUTOFFS = [360, 340, 320, 300]     # 공통 목표 에너지선 sweep


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"]


def run(method, ini, Y, B, u_0, a0, n, BtB, P, Lc):
    np.random.seed(0); theta = ini.copy(); ths = np.zeros((T, n)); alpha = a0
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
    ess = float(np.nanmedian(ess_per_node(ths[BURN:])))
    return U_tr, ess


def first_passage(U_tr, cut):
    idx = np.where(U_tr <= cut)[0]
    return int(idx[0]) if len(idx) else None


def main():
    METHODS = ["AWSGLD", "cycSGLD"]
    reach = {m: {c: [] for c in CUTOFFS} for m in METHODS}
    ess = {m: [] for m in METHODS}
    t0 = time.time()
    print(f"cut-off 도달시간 + ESS | AWSGLD vs cycSGLD | seeds={SEEDS} | cutoffs={CUTOFFS}\n", flush=True)
    for s in SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
        a0 = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
        for m in METHODS:
            U_tr, e = run(m, ini, Y, B, u_0, a0, n, BtB, P, Lc)
            ess[m].append(e)
            for c in CUTOFFS:
                fp = first_passage(U_tr, c); reach[m][c].append(fp)
        pr = " | ".join(f"{m}: " + ",".join(str(reach[m][c][-1]) for c in CUTOFFS) + f" ESS={ess[m][-1]:.1f}" for m in METHODS)
        print(f"seed {s} | {pr}", flush=True)

    def mean_reach(m, c):
        v = [x for x in reach[m][c] if x is not None]
        return (np.mean(v), len(v)) if v else (None, 0)

    print("\n=== cut-off 별 평균 도달 반복수 (도달 seed수/5) ===")
    print(f"{'cutoff':>8} | " + " | ".join(f"{m:>18}" for m in METHODS))
    for c in CUTOFFS:
        cells = []
        for m in METHODS:
            mr, k = mean_reach(m, c)
            cells.append(f"{mr:.0f} ({k}/5)" if mr is not None else "미도달 (0/5)")
        print(f"{c:>8} | " + " | ".join(f"{c2:>18}" for c2 in cells))
    print(f"\nESS median | AWSGLD={np.mean(ess['AWSGLD']):.1f}  cycSGLD={np.mean(ess['cycSGLD']):.1f}  ({int(time.time()-t0)}s)")

    with open(os.path.join(HERE, "reach_time_ess.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "ess"] + [f"reach@{c}" for c in CUTOFFS])
        for m in METHODS:
            for i, s in enumerate(SEEDS):
                w.writerow([m, s, round(ess[m][i], 2)] + [reach[m][c][i] for c in CUTOFFS])

    # 그림: (a) cut-off별 도달 반복수 (그룹 막대), (b) ESS
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]; x = np.arange(len(CUTOFFS)); wd = 0.36
    for j, m in enumerate(METHODS):
        means = [mean_reach(m, c)[0] or 0 for c in CUTOFFS]
        ax.bar(x + (j - 0.5) * wd, means, wd, color=COL[m], alpha=0.85, label=m)
    ax.set_xticks(x); ax.set_xticklabels([f"U≤{c}" for c in CUTOFFS])
    ax.set_xlabel("cut-off energy level"); ax.set_ylabel("iterations to reach (mean)")
    ax.set_title("(a) time to reach common cut-off\n(둘 다 도달 → 비슷)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.2)
    ax = axes[1]; rng = np.random.RandomState(0)
    for i, m in enumerate(METHODS):
        v = np.array(ess[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.25
        ax.scatter(xs, v, s=48, color=COL[m], alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=COL[m], lw=3, zorder=4)
        ax.text(i, v.mean(), f"{v.mean():.1f}", ha="center", va="bottom", fontsize=11, color=COL[m], fontweight="bold")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=10)
    ax.set_title("(b) ESS median\n(여기서 AWSGLD 우위)", fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.2)
    fig.suptitle("BSS posterior: reaching a common cut-off is similar — the real gap is ESS",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "reach_time_ess.png"), dpi=140, bbox_inches="tight")
    print("저장: reach_time_ess.png, reach_time_ess.csv")


if __name__ == "__main__":
    main()
