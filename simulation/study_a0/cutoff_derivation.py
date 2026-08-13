"""cut-off = 312 의 도출 근거 시각화 (지표 + 수치).

논리
----
cut-off 는 '수렴한 샘플러가 정착하는 에너지(typical set)'로 정의한다. 수렴 판정은 표준
Gelman-Rubin R̂ (< 1.2). 과분산 3-chain(μ_N/μ_W/μ_S)으로:
  (a) running R̂max 곡선 → 처음 1.2 아래로 내려가는 반복수 = 수렴 시점
  (b) 그 이후 정상상태 에너지 median = cut-off
AWSGLD 만 R̂<1.2 로 수렴 → cut-off 자격. cycSGLD 는 R̂≈3.2 로 미수렴.

3 seed 로 수치 집계, seed0 곡선으로 그림. 출력: cutoff_derivation.png / .csv
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

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5000; BURN = 500; BATCH = 100; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
CYC = 10; SEEDS = [0, 1, 2]; RHAT_THR = 1.2
INITS = [("μ_N", -0.8), ("μ_W", 1.0), ("μ_S", 2.5)]
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
    return ths, E.energy_trace_common(ths, Y, B, u_0, a0)


def rhat_max(chains, lo, hi):
    arrs = np.stack([c[lo:hi] for c in chains], 0); M, L, n = arrs.shape
    cm = arrs.mean(1); gm = cm.mean(0)
    B_over_L = ((cm - gm) ** 2).sum(0) / (M - 1); W = arrs.var(1, ddof=1).mean(0)
    R = np.sqrt(np.clip(((L - 1) / L * W + B_over_L) / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R))


def main():
    checkpoints = np.unique(np.linspace(BURN + 100, T, 25).astype(int))
    agg = {m: {"conv": [], "cut": []} for m in COL}
    demo = {}
    t0 = time.time()
    for s in SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
        for m in COL:
            chains = []; Utrs = []
            for ci, (nm, val) in enumerate(INITS):
                ths, U = run(m, np.full(n, float(val)), Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * s + ci)
                chains.append(ths); Utrs.append(U)
            rr = np.array([rhat_max(chains, BURN, tc) for tc in checkpoints])
            below = np.where(rr < RHAT_THR)[0]
            conv = int(checkpoints[below[0]]) if len(below) else None
            ref = conv if conv else BURN
            cut = float(np.median(np.concatenate([U[ref:] for U in Utrs])))
            agg[m]["conv"].append(conv); agg[m]["cut"].append(cut)
            if s == 0: demo[m] = (Utrs, rr, conv, cut)
        print(f"  seed {s} done ({int(time.time()-t0)}s)", flush=True)

    aw_cut = [c for c in agg["AWSGLD"]["cut"]]
    aw_conv = [c for c in agg["AWSGLD"]["conv"] if c]
    CUT = float(np.mean(aw_cut)); CUT_SD = float(np.std(aw_cut))
    print("\n=== cut-off 도출 수치 ===")
    print(f"AWSGLD  R̂<1.2 수렴 반복수: {agg['AWSGLD']['conv']}  (평균 {np.mean(aw_conv):.0f})")
    print(f"AWSGLD  수렴후 정상상태 에너지(=cut-off): {[round(x,1) for x in aw_cut]}  → {CUT:.1f} ± {CUT_SD:.1f}")
    print(f"cycSGLD R̂<1.2 수렴: {agg['cycSGLD']['conv']} (미수렴)  에너지 {[round(x,1) for x in agg['cycSGLD']['cut']]}")

    with open(os.path.join(HERE, "cutoff_derivation.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "rhat_conv_iter", "stationary_energy_after_conv"])
        for m in COL:
            for i, s in enumerate(SEEDS):
                w.writerow([m, s, agg[m]["conv"][i], round(agg[m]["cut"][i], 1)])

    # ── 도출 그림 (seed0) ──
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4))
    # (a) running R̂
    ax = axes[0]
    for m in COL:
        _, rr, conv, cut = demo[m]
        ax.plot(checkpoints, rr, "o-", color=COL[m], lw=1.8, ms=4, label=m)
    ax.axhline(RHAT_THR, color="red", ls="--", lw=1.3, alpha=0.8, label="R̂ = 1.2 (수렴 기준)")
    convA = demo["AWSGLD"][2]
    if convA:
        ax.axvline(convA, color=COL["AWSGLD"], ls=":", lw=1.5, alpha=0.8)
        ax.annotate(f"AWSGLD 수렴\n@ {convA} iter", (convA, RHAT_THR),
                    xytext=(convA - 1700, 2.0), fontsize=9, color=COL["AWSGLD"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=COL["AWSGLD"]))
    ax.annotate("cycSGLD: R̂≈3.2, 끝까지 미수렴", (checkpoints[-1], demo["cycSGLD"][1][-1]),
                xytext=(1200, 3.5), fontsize=9, color=COL["cycSGLD"], fontweight="bold")
    ax.set_xlabel("iteration up to"); ax.set_ylabel("running R̂ max (over 400 nodes)")
    ax.set_title("(a) 수렴 판정: running R̂ (Gelman-Rubin)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.2)
    # (b) 에너지 trace + cut-off
    ax = axes[1]
    Utrs, _, convA, cutA = demo["AWSGLD"]
    for U in Utrs: ax.plot(U, color=COL["AWSGLD"], lw=0.5, alpha=0.5)
    if convA: ax.axvline(convA, color=COL["AWSGLD"], ls=":", lw=1.5, alpha=0.8, label=f"수렴 @ {convA} iter")
    ax.axhspan(CUT - CUT_SD, CUT + CUT_SD, color="#C0392B", alpha=0.15)
    ax.axhline(CUT, color="#C0392B", ls="--", lw=1.6, label=f"cut-off = {CUT:.0f} ± {CUT_SD:.0f}")
    ax.annotate(f"cut-off = 수렴후 정상상태 에너지 median\n= {CUT:.1f} ± {CUT_SD:.1f}",
                (T * 0.52, CUT), xytext=(T * 0.30, CUT + 55), fontsize=9, color="#C0392B", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C0392B"))
    ax.set_xlabel("iteration"); ax.set_ylabel("U(theta) — 공통 에너지")
    ax.set_title("(b) cut-off = 수렴후 정상상태 에너지", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.2)
    fig.suptitle(f"Cut-off 도출: R̂<1.2 로 수렴한 AWSGLD 의 정상상태 에너지 = {CUT:.0f} ± {CUT_SD:.0f}  "
                 f"(cycSGLD 는 미수렴, cut-off 자격 없음)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "cutoff_derivation.png"), dpi=140, bbox_inches="tight")
    print(f"\n저장: cutoff_derivation.png, cutoff_derivation.csv (cut-off={CUT:.0f}±{CUT_SD:.0f}, {int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
