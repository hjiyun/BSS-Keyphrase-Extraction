"""통합 종합표 — 전 지표를 4-chain × seed0 단일 setup에서 계산.
Spearman/MSE/NDCG@50 | split-R̂ (median/q95/max) | ESS | Lowest U / Reached.
cutoff = 이 setup 에서 AWSGLD 정상상태 에너지(post-burn median, 4체인 median).
AWSGLD 는 R̂·복원에 π-가중, 나머지 raw. 에너지(min U)는 전부 raw 궤적.
출력: unified_n100.csv + 콘솔.
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
import data_generator as DG
from local_trap_landscape import PARAMS
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100; E.T = 20000; E.BURN = 2000; E.BATCH = 50
T = E.T; BURN = E.BURN; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SEED = 0; INITS = [-0.8, 1.0, 2.5, "rand"]
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]


def gen(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, ts


def run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); Js = np.full(T, -1); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 40)); es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        eps = 0.3 / ((t + 1) ** 0.6 + 10)
        U = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2)
        bidx = np.random.choice(n, E.BATCH, replace=False) if E.BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, alpha, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        gm = 1.0
        if t < warm:
            es.append(U)
            if t == warm - 1:
                lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
        else:
            J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
            gm = float(np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; Js[t] = J; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths, Js, aw


def ndcg_at_k(ts, th, k):
    rel = np.argsort(np.argsort(ts)).astype(float) / max(len(ts) - 1, 1)
    pred = np.argsort(th)[::-1][:k]; ideal = np.argsort(ts)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    d = np.sum((2 ** rel[pred] - 1) * disc); i = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(d / i) if i > 0 else 0.0


def split_rhat_coords(posts, wts):
    hm = []; hv = []; hL = None
    for ci, post in enumerate(posts):
        L = post.shape[0]; h = L // 2; hL = h
        for sl in (slice(0, h), slice(h, 2 * h)):
            seg = post[sl]
            if wts[ci] is not None:
                w = wts[ci][sl]; sw = w.sum()
                m = (w[:, None] * seg).sum(0) / sw; v = (w[:, None] * (seg - m) ** 2).sum(0) / sw
            else:
                m = seg.mean(0); v = seg.var(0, ddof=1)
            hm.append(m); hv.append(v)
    hm = np.array(hm); hv = np.array(hv); M2 = len(hm)
    gm = hm.mean(0); Bo = ((hm - gm) ** 2).sum(0) / (M2 - 1); W = hv.mean(0)
    return np.sqrt(np.clip(((hL - 1) / hL * W + Bo) / np.maximum(W, 1e-12), 0, None))


def main():
    graph, Y, B, u_0, ts = gen(N, SEED); n = N
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
    D = {}; t0 = time.time()
    print(f"통합 4-chain×seed0 | n={N} T={T}\n", flush=True)
    for m in METHODS:
        posts = []; wts = []; ess_list = []; minU = []; statU = []; pn = 0.0; pd = 0.0; kong = None
        for ci, val in enumerate(INITS):
            ini = (np.random.RandomState(7000 + 100 * SEED + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))
            if m == "AWSGLD":
                ths, J, aw = run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * SEED + ci)
                post = ths[BURN:]; Jp = J[BURN:]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
                posts.append(post); wts.append(w); pn += (w[:, None] * post).sum(0); pd += sw
                if ci == 0: kong = (sw ** 2) / (np.sum(w ** 2) + 1e-300)
            else:
                np.random.seed(100 * SEED + ci)
                ths = E.run_acmh(graph, Y, B, u_0, ini, a0, 100 * SEED + ci) if m == "acMH" \
                    else E.run_sgld_family(m, graph, Y, B, u_0, ini, a0, 100 * SEED + ci)
                post = ths[BURN:]; posts.append(post); wts.append(None); pn += post.sum(0); pd += post.shape[0]
            ess_list.append(np.nanmedian(ess_per_node(post)))
            Utr = E.energy_trace_common(post, Y, B, u_0, a0)
            minU.append(float(Utr.min())); statU.append(float(np.median(Utr)))
        th = pn / pd
        R = split_rhat_coords(posts, wts)
        D[m] = dict(sp=spearmanr(ts, th).statistic, mse=float(np.mean((th - ts) ** 2)),
                    ndcg=ndcg_at_k(ts, th, 50), rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)),
                    rmax=float(np.nanmax(R)), ess=float(np.median(ess_list)), kong=kong,
                    minU=minU, statU=statU, ess_chain=[float(e) for e in ess_list])
        print(f"  {m:>8}: Spear {D[m]['sp']:.2f} R̂ med/q95/max {D[m]['rmed']:.2f}/{D[m]['rq95']:.2f}/{D[m]['rmax']:.2f} "
              f"ESS {D[m]['ess']:.0f} minU {np.mean(minU):.0f}  ({int(time.time()-t0)}s)", flush=True)

    CUT = float(np.round(np.median(D["AWSGLD"]["statU"])))   # cutoff = AWSGLD 정상상태 에너지 (4체인 median)
    NC = len(INITS)
    print(f"\ncutoff = {CUT:.0f} (AWSGLD 정상상태 에너지, 4체인 median)\n")
    hdr = f"{'Sampler':>8} {'Spear':>6} {'MSE':>6} {'NDCG50':>7} | {'R̂med':>6} {'R̂q95':>6} {'R̂max':>6} | {'ESS':>5} | {'LowU':>5} {'Reach':>6}"
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        d = D[m]; low = [max(CUT, v) for v in d["minU"]]; reach = sum(1 for v in d["minU"] if v <= CUT)
        d["low"] = float(np.mean(low)); d["reach"] = reach
        print(f"{m:>8} {d['sp']:>6.2f} {d['mse']:>6.2f} {d['ndcg']:>7.3f} | {d['rmed']:>6.3f} {d['rq95']:>6.3f} {d['rmax']:>6.3f} | "
              f"{d['ess']:>5.0f} | {d['low']:>5.0f} {reach:>4}/{NC}")

    with open(os.path.join(HERE, "unified_n100.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "spearman", "mse", "ndcg50", "rhat_median", "rhat_q95", "rhat_max",
                                        "ess", "kong_ess", "lowest_U", "reached", "n_chains", "cutoff"])
        for m in METHODS:
            d = D[m]
            w.writerow([m, round(d["sp"], 4), round(d["mse"], 4), round(d["ndcg"], 4), round(d["rmed"], 4),
                        round(d["rq95"], 4), round(d["rmax"], 4), round(d["ess"], 2),
                        ("" if d["kong"] is None else round(d["kong"], 1)), round(d["low"], 1), d["reach"], NC, CUT])
    # 연쇄별 detail (strip 그림용): 연쇄마다 clamped min_U, ESS
    with open(os.path.join(HERE, "unified_detail_n100.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "chain", "min_U", "lowest_U_clamped", "ess", "cutoff"])
        for m in METHODS:
            d = D[m]
            for c in range(NC):
                w.writerow([m, c, round(d["minU"][c], 2), round(max(CUT, d["minU"][c]), 2),
                            round(d["ess_chain"][c], 2), CUT])
    print(f"\n저장: unified_n100.csv, unified_detail_n100.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
