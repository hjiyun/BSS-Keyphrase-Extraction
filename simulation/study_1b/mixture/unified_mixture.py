"""합성 mixture 에너지 위 6 샘플러 통합표 — study_a0 지표 세트와 동일.

Spearman/MSE/NDCG@50 (θ* 기준) | split-R̂ (median/q95/max, 4연쇄) | ESS(4연쇄 mean)
| Lowest U / Reached (cutoff = AWSGLD 정상상태 U_mix). + mode 방문수(다봉 진단).
AWSGLD 는 R̂·복원에 π-가중(w=G[J]), 나머지 raw. 에너지·mode 는 전부 raw 궤적.
샘플러는 전부 mixture_energy.MixtureEnergy 를 항해한다 (실제 다봉).

출력: unified_mixture.csv + unified_mixture_detail.csv + 콘솔.
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "study_a0", "_archive"))
import mixture_energy as MX                          # noqa: E402
import keyphrase_functions_awsgld as kfa            # noqa: E402
from extra_metrics import ess_per_node               # study_a0/_archive  # noqa: E402

# ── 설정 (study_a0 unified 규약) ──
N = 100; T = 8000; BURN = 1000; BATCH = 50
SEED = 0; INITS = [-0.8, 1.0, 2.5, "rand"]           # μ_N / μ_W / μ_S / random (a0 동일)
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
# SGLD 계열 하이퍼 (study_a0 energy_diagnostics 와 동일)
SGLD_LR = 0.02; QSGLD_LR = 0.3; CYCSGLD_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FRICTION = 0.1


def precond(B, n):
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    return BtB, P, L


def run_acmh(EN, ini, n, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n))
    BtB_inv = np.linalg.solve(EN.BtB + 1e-8 * np.eye(n), np.eye(n))
    cov = BtB_inv * EN.sig2 * 4.0 / n; Lc = np.linalg.cholesky(cov + 1e-10 * np.eye(n))
    Uc = EN.energy(theta)
    for t in range(T):
        star = theta + Lc @ np.random.randn(n)
        Us = EN.energy(star)
        if np.log(np.random.rand() + 1e-300) < (-Us + Uc):
            theta = star; Uc = Us
        ths[t] = theta
    return ths


def run_sgld_family(method, EN, ini, n, P, L, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); v = np.zeros(n)
    for t in range(T):
        bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
        gU = EN.grad(theta, batch_idx=bidx)
        if method == "SGLD":
            eps = SGLD_LR / ((t + 1) ** 0.6 + 10)
            theta = theta - eps * gU + np.sqrt(2 * TAU * eps) * np.random.randn(n)
        elif method == "qSGLD":
            eps = QSGLD_LR / ((t + 1) ** 0.6 + 10)
            theta = theta - eps * (P @ gU) + np.sqrt(2 * TAU * eps) * (L @ np.random.randn(n))
        elif method == "cycSGLD":
            cl = max(1, T // CYC_CYCLES); beta = (t % cl) / cl
            eps = CYCSGLD_LR / 2 * (np.cos(np.pi * min(beta, 0.8)) + 1)
            tk = TAU if beta >= 0.8 else TAU / 1e4
            theta = theta - eps * gU + np.sqrt(2 * tk * eps) * np.random.randn(n)
        elif method == "SGHMC":
            eta = SGHMC_LR / ((t + 1) ** 0.6 + 10); theta = theta + v
            v = (1 - SGHMC_FRICTION) * v - eta * gU + np.sqrt(2 * SGHMC_FRICTION * eta * TAU) * np.random.randn(n)
        theta = np.clip(theta, -700, 700); ths[t] = theta
    return ths


def run_awsgld(EN, ini, n, P, L, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); Js = np.full(T, -1)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 40)); es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        eps = 0.3 / ((t + 1) ** 0.6 + 10)
        U = EN.energy(theta)
        bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
        gU = EN.grad(theta, batch_idx=bidx)
        gm = 1.0
        if t < warm:
            es.append(U)
            if t == warm - 1:
                lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
        else:
            J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
            gm = float(np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (L @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; Js[t] = J
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
    graph, Y, B, u_0, ts, z = MX.gen(N, SEED); n = N
    centers, kw, nonkw, decoys = MX.build_centers(z, seed=0)
    EN = MX.MixtureEnergy(Y, B, centers)
    BtB, P, L = precond(B, n)
    D = {}; t0 = time.time()
    print(f"합성 mixture 4-chain×seed0 | n={N} T={T} K={MX.K_MODES} | 키워드(S∪W)={len(kw)} 미끼(N)={len(nonkw)}\n", flush=True)
    for m in METHODS:
        posts = []; wts = []; ess_list = []; minU = []; statU = []; nmodes = []
        pn = np.zeros(n); pd = 0.0; kong = None
        for ci, val in enumerate(INITS):
            ini = (np.random.RandomState(7000 + 100 * SEED + ci).randn(n) * 1.5 if val == "rand"
                   else np.full(n, float(val)))
            sd = 100 * SEED + ci
            if m == "AWSGLD":
                ths, Js, aw = run_awsgld(EN, ini, n, P, L, seed=sd)
                post = ths[BURN:]; Jp = Js[BURN:]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
                posts.append(post); wts.append(w); pn += (w[:, None] * post).sum(0); pd += sw
                if ci == 0: kong = (sw ** 2) / (np.sum(w ** 2) + 1e-300)
            elif m == "acMH":
                ths = run_acmh(EN, ini, n, sd); post = ths[BURN:]
                posts.append(post); wts.append(None); pn += post.sum(0); pd += post.shape[0]
            else:
                ths = run_sgld_family(m, EN, ini, n, P, L, sd); post = ths[BURN:]
                posts.append(post); wts.append(None); pn += post.sum(0); pd += post.shape[0]
            ess_list.append(float(np.nanmedian(ess_per_node(post))))
            Utr = EN.energy_trace(post)
            minU.append(float(Utr.min())); statU.append(float(np.median(Utr)))
            mo = np.array([EN.mode(post[i]) for i in range(0, post.shape[0], 5)])
            nmodes.append(int(len(set(mo.tolist()))))
        th = pn / pd
        R = split_rhat_coords(posts, wts)
        D[m] = dict(sp=spearmanr(ts, th).statistic, mse=float(np.mean((th - ts) ** 2)),
                    ndcg=ndcg_at_k(ts, th, 50), rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)),
                    rmax=float(np.nanmax(R)), ess=float(np.mean(ess_list)), kong=kong,
                    minU=minU, statU=statU, ess_chain=[float(e) for e in ess_list], nmodes=nmodes)
        print(f"  {m:>8}: Spear {D[m]['sp']:.2f} MSE {D[m]['mse']:.2f} NDCG {D[m]['ndcg']:.3f} "
              f"R̂ {D[m]['rmed']:.2f}/{D[m]['rq95']:.2f}/{D[m]['rmax']:.2f} ESS {D[m]['ess']:.0f} "
              f"modes {np.mean(nmodes):.1f}/{MX.K_MODES}  ({int(time.time()-t0)}s)", flush=True)

    CUT = float(np.round(np.median(D["AWSGLD"]["statU"])))
    NC = len(INITS)
    print(f"\ncutoff = {CUT:.0f} (AWSGLD 정상상태 U_mix, 4연쇄 median)\n")
    hdr = (f"{'Sampler':>8} {'Spear':>6} {'MSE':>6} {'NDCG50':>7} | {'R̂med':>6} {'R̂q95':>6} {'R̂max':>6} | "
           f"{'ESS':>5} | {'LowU':>6} {'Reach':>6} | {'modes':>6}")
    print(hdr); print("-" * len(hdr))
    for m in METHODS:
        d = D[m]; low = [max(CUT, v) for v in d["minU"]]; reach = sum(1 for v in d["minU"] if v <= CUT)
        d["low"] = float(np.mean(low)); d["reach"] = reach
        print(f"{m:>8} {d['sp']:>6.2f} {d['mse']:>6.2f} {d['ndcg']:>7.3f} | {d['rmed']:>6.3f} {d['rq95']:>6.3f} {d['rmax']:>6.3f} | "
              f"{d['ess']:>5.0f} | {d['low']:>6.0f} {reach:>4}/{NC} | {np.mean(d['nmodes']):>5.1f}")

    with open(os.path.join(_HERE, "unified_mixture.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "spearman", "mse", "ndcg50", "rhat_median", "rhat_q95",
                                        "rhat_max", "ess", "kong_ess", "lowest_U", "reached", "n_chains",
                                        "cutoff", "modes_visited_mean"])
        for m in METHODS:
            d = D[m]
            w.writerow([m, round(d["sp"], 4), round(d["mse"], 4), round(d["ndcg"], 4), round(d["rmed"], 4),
                        round(d["rq95"], 4), round(d["rmax"], 4), round(d["ess"], 2),
                        ("" if d["kong"] is None else round(d["kong"], 1)), round(d["low"], 1), d["reach"], NC,
                        CUT, round(float(np.mean(d["nmodes"])), 2)])
    with open(os.path.join(_HERE, "unified_mixture_detail.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "chain", "min_U", "lowest_U_clamped", "ess", "modes", "cutoff"])
        for m in METHODS:
            d = D[m]
            for c in range(NC):
                w.writerow([m, c, round(d["minU"][c], 2), round(max(CUT, d["minU"][c]), 2),
                            round(d["ess_chain"][c], 2), d["nmodes"][c], CUT])
    print(f"\n저장: unified_mixture.csv, unified_mixture_detail.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
