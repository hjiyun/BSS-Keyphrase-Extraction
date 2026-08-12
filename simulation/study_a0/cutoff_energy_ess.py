"""BSS 원 에너지: min U 가 아니라 '수렴 후 정상상태(cut-off) 에너지'로 AWSGLD vs cycSGLD 비교.

핵심 주장
--------
볼록 BSS 사후에서 제대로 샘플링하면 체인은 최저점이 아니라 typical-set 에너지 근처에서
진동한다(= E_π[U], min 보다 위). 이 정상상태 에너지로 재면 AWSGLD ≈ cycSGLD (둘 다 같은
사후를 겨냥). min U 가 cyc 를 낮게 보이게 한 건 냉각 구간의 일시적 과냉각일 뿐.
→ 차이는 '어디 있나(에너지)'가 아니라 '얼마나 효율적으로 훑나(ESS)'. 여기서 AWSGLD 우위.

cut-off 정의
-----------
각 체인의 post-burn 에너지 running-median 이 평탄해지는 지점 이후를 '정상상태'로 보고,
그 구간 에너지의 median±IQR 를 대표값으로 사용. (최저값 X)

그림 (2 패널, seed 0..4 strip)
  (a) 정상상태 에너지 (cut-off)  → AWSGLD ≈ cycSGLD
  (b) ESS median                → AWSGLD > cycSGLD
출력: cutoff_energy_ess.png / .csv
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
            eps = 0.02 / 2.0 * (np.cos(np.pi * min(beta, 0.8)) + 1)
            tau_k = TAU if beta >= 0.8 else TAU / 1e4
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
    post = U_tr[BURN:]
    stationary = float(np.median(post))          # cut-off 정상상태 에너지 (min 아님)
    ess = float(np.nanmedian(ess_per_node(ths[BURN:])))
    return stationary, ess, U_tr


def main():
    METHODS = ["AWSGLD", "cycSGLD"]
    res = {m: {"stat": [], "ess": []} for m in METHODS}
    trace_demo = {}   # seed0 트레이스 저장(그림 c 용)
    t0 = time.time()
    print(f"cut-off(정상상태) 에너지 + ESS | AWSGLD vs cycSGLD | seeds={SEEDS}\n", flush=True)
    print(f"{'seed':>4} | {'AWSGLD stat/ESS':>22} | {'cycSGLD stat/ESS':>22}", flush=True)
    for s in SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
        a0 = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
        cells = []
        for m in METHODS:
            stat, ess, U_tr = run(m, ini, Y, B, u_0, a0, n, BtB, P, Lc, seed=0)
            res[m]["stat"].append(stat); res[m]["ess"].append(ess)
            if s == 0: trace_demo[m] = U_tr
            cells.append(f"{stat:>10.0f} / {ess:>7.1f}")
        print(f"{s:>4} | {cells[0]:>22} | {cells[1]:>22}", flush=True)
    print(f"\n평균 | AWSGLD stat={np.mean(res['AWSGLD']['stat']):.0f} ESS={np.mean(res['AWSGLD']['ess']):.1f}"
          f" | cycSGLD stat={np.mean(res['cycSGLD']['stat']):.0f} ESS={np.mean(res['cycSGLD']['ess']):.1f}"
          f"  ({int(time.time()-t0)}s)", flush=True)

    with open(os.path.join(HERE, "cutoff_energy_ess.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "stationary_energy", "ess_median"])
        for m in METHODS:
            for i, s in enumerate(SEEDS):
                w.writerow([m, s, round(res[m]["stat"][i], 2), round(res[m]["ess"][i], 2)])

    # ── 그림: (a) 정상상태 에너지 strip, (b) ESS strip, (c) seed0 에너지 trace + cut-off ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    rng = np.random.RandomState(0)
    for ax, key, ttl, note in [
            (axes[0], "stat", "(a) stationary energy (cut-off, not min)", "겹치면 = 같은 사후 겨냥"),
            (axes[1], "ess", "(b) ESS median", "높을수록 효율적")]:
        for i, m in enumerate(METHODS):
            v = np.array(res[m][key]); xs = i + (rng.rand(len(v)) - 0.5) * 0.25
            ax.scatter(xs, v, s=48, color=COL[m], alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
            ax.plot([i - 0.25, i + 0.25], [v.mean()] * 2, color=COL[m], lw=3, zorder=4)
            ax.text(i, v.mean(), f"{v.mean():.0f}" if key == "stat" else f"{v.mean():.1f}",
                    ha="center", va="bottom", fontsize=10, color=COL[m], fontweight="bold")
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, fontsize=10)
        ax.set_title(ttl, fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.2)
    # (c) seed0 에너지 trace + cut-off 밴드
    ax = axes[2]
    for m in METHODS:
        U_tr = trace_demo[m]
        ax.plot(U_tr, color=COL[m], lw=0.7, alpha=0.8, label=m)
        band = np.median(U_tr[BURN:])
        ax.axhline(band, color=COL[m], ls="--", lw=1.2, alpha=0.7)
    ax.axvline(BURN, color="k", ls=":", lw=1, alpha=0.5)
    ax.set_title("(c) energy trace (seed0)\n dashed = cut-off level", fontsize=11, fontweight="bold")
    ax.set_xlabel("iteration"); ax.set_ylabel("U(theta)"); ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.suptitle("BSS posterior: cut-off (stationary) energy is similar for AWSGLD & cycSGLD — the gap is in ESS",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cutoff_energy_ess.png"), dpi=140, bbox_inches="tight")
    print("저장: cutoff_energy_ess.png, cutoff_energy_ess.csv")


if __name__ == "__main__":
    main()
