"""재설계 시뮬레이션 GPU판 — 대형 n(5000·10000). 표준 BSS(단봉), full 프로토콜.
- torch dense matvec(GPU). 샘플러: SGLD/qSGLD/cycSGLD/SGHMC/AWSGLD(θ·α·σ² Langevin) + tMH(논문 MH, 벡터화).
  (componentwise acMH는 대형 n 부적합 → 논문의 tMH within Gibbs로 대체)
- 지표: γ별 FDR(P/R/F1/realFDR) / 순위(Spearman·Kendall·Top-k·NDCG·MSE·AUC) / σ̂·Jaccard·cov90. 전부 θ*·raw.
사용: python3 redesign_gpu.py <n_seed> <n> <mu_gap> <alpha> [zeta] [kdoc] [sparse]
"""
import os, sys, csv, itertools
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
MUGAP = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
ZETA  = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
KDOC  = int(sys.argv[6]) if len(sys.argv) > 6 else 1
SPARSE = int(sys.argv[7]) if len(sys.argv) > 7 else 0
PCROSS = 0.004; SIG = 0.4; PIN, POUT = 0.30, 0.02
METH = ["tMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
GAMMAS = [0.05, 0.1, 0.15, 0.2, 0.3]
T = 5000; BURN = 2500; THIN = 5; M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MUGAP, 0.0, -MUGAP]); ts = mu[z] + SIG * rng.standard_normal(N)
    doc = rng.integers(0, KDOC, N) if KDOC > 1 else np.zeros(N, int)
    A = np.zeros((N, N), np.float64)
    for i in range(N):
        pr = np.where(doc == doc[i], np.where(z == z[i], PIN, POUT), PCROSS)
        r = rng.random(N); r[:i + 1] = 1.0
        A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    deg = A.sum(1); keep = deg > 0; A = A[np.ix_(keep, keep)]; ts = ts[keep]; z = z[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u_0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    if SPARSE:
        pos = np.where(Y == 1)[0]; Y[pos[rng.random(len(pos)) < 0.75]] = 0.0
    a0 = kfa.alpha_find(u_0, Y, kfa.grid)
    truth = np.where(ts > 0)[0]
    return n, B, u_0, ts, z, Y, float(a0), truth


def run_seed(seed):
    n, Bn, u0n, ts, z, Yn, a0, truth = gen(seed)
    B = torch.tensor(Bn, device=DEV); u_0 = torch.tensor(u0n, device=DEV); Y = torch.tensor(Yn, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
    grid = torch.tensor(kfa.grid, device=DEV)

    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def gradU(th, alpha, s2):
        pi = sig(th); temp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10); denom = torch.clamp(1 - temp, min=1e-10)
        gll = torch.where(Y == 1, 1 - pi, -(1 - alpha) * pi * (1 - pi) / denom)   # d loglik/dθ
        return -gll + (BtB @ (th - u_0)) / s2
    def energyU(th, alpha, s2):
        pi = sig(th); temp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(temp) + (1 - Y) * torch.log(1 - temp)).sum(); Bd = B @ (th - u_0)
        return -ll + (Bd @ Bd) / (2 * s2)
    def s2_gibbs(th):
        C = float(((B @ (th - u_0)) ** 2).sum());
        return float(np.clip(1.0 / np.random.gamma(n / 2 + 0.001, 1.0 / (C / 2 + 0.001)), 0.05, 100.0))
    def alpha_eb(th):
        pi = sig(th); temp = torch.clamp((1 - grid[:, None]) * pi[None, :], 1e-10, 1 - 1e-10)
        lk = (Y[None, :] * torch.log(temp) + (1 - Y)[None, :] * torch.log(1 - temp)).sum(1)
        return grid[torch.argmax(lk)]

    g = torch.Generator(device=DEV)
    def sampler(method, seed_ci):
        g.manual_seed(seed_ci); theta = torch.randn(n, generator=g, device=DEV) * 1.5
        aa = torch.log(torch.tensor(a0 / (1 - a0), device=DEV)); ss = torch.zeros((), device=DEV); v = torch.zeros(n, device=DEV)
        aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
        keep = []; wts = []
        for t in range(T):
            alpha = torch.sigmoid(aa); s2 = torch.exp(torch.clamp(ss, -6, 6))
            if method == "tMH":                                   # 논문 tMH: RW 제안 N(θ, P·σ²·b/n) + MH
                s2v = s2_gibbs(theta); a_eb = alpha_eb(theta)
                prop = theta + np.sqrt(4.0 / n) * torch.sqrt(torch.tensor(s2v)) * (Lc @ torch.randn(n, generator=g, device=DEV))
                dE = -energyU(prop, a_eb, s2v) + energyU(theta, a_eb, s2v)
                if torch.log(torch.rand((), generator=g, device=DEV)) < dE: theta = prop
                if t >= BURN and t % THIN == 0: keep.append(theta.clone()); wts.append(1.0)
                continue
            gth = gradU(theta, alpha, s2)
            if method == "SGLD":
                ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gth + np.sqrt(2 * ek) * torch.randn(n, generator=g, device=DEV)
            elif method == "qSGLD":
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gth) + np.sqrt(2 * ek) * (Lc @ torch.randn(n, generator=g, device=DEV))
            elif method == "cycSGLD":
                cl = max(1, T // 10); be = (t % cl) / cl; ek = 0.01 / 2 * (np.cos(np.pi * min(be, 0.8)) + 1); tk = 1.0 if be >= 0.8 else 1e-4
                theta = theta - ek * gth + np.sqrt(2 * tk * ek) * torch.randn(n, generator=g, device=DEV)
            elif method == "SGHMC":
                eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v
                v = 0.9 * v - eta * gth + np.sqrt(2 * 0.1 * eta) * torch.randn(n, generator=g, device=DEV)
            elif method == "AWSGLD":
                Uv = float(energyU(theta, alpha, s2)); gm = 1.0
                if not np.isfinite(Uv): Uv = float(emin) if emin is not None else 0.0
                if t < warm:
                    es.append(Uv)
                    if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
                else:
                    J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
                ek = 0.3 / ((t + 1) ** 0.6 + 10)
                theta = theta - ek * gm * (P @ gth) + np.sqrt(2 * ek * gm) * (Lc @ torch.randn(n, generator=g, device=DEV))
                if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
            theta = torch.clamp(theta, -700, 700)
            # α, σ² Langevin (재매개변수화)
            pi = sig(theta); tt = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
            gA = ((Y / (1 - alpha) - (1 - Y) * pi / (1 - tt)).sum() - 1 / alpha + 1 / (1 - alpha)) * alpha * (1 - alpha)
            C = ((B @ (theta - u_0)) ** 2).sum(); gS = -(C / 2 + 0.001) / s2 + (n / 2 + 0.001)
            eA = 0.2 * (100.0 / n) / ((t + 1) ** 0.6 + 10); eS = 0.2 * (100.0 / n) / ((t + 1) ** 0.6 + 10)  # 스칼라 Langevin step n-스케일(gA,gS~O(n) 발산 방지)
            aa = torch.clamp(aa - eA * gA + np.sqrt(2 * eA) * torch.randn((), generator=g, device=DEV), -6, 6)
            ss = torch.clamp(ss - eS * gS + np.sqrt(2 * eS) * torch.randn((), generator=g, device=DEV), -6, 6)
            if t >= BURN and t % THIN == 0: keep.append(theta.clone()); wts.append(float(aw[J]) if method == "AWSGLD" else 1.0)
        S = torch.stack(keep); w = torch.tensor(wts, device=DEV)
        return S, w   # (Skeep, n), (Skeep,)

    # 그룹(θ* 평균 내림차순 S/W/N), truth
    uz = sorted(np.unique(z), key=lambda gv: ts[z == gv].mean(), reverse=True)
    grp = {i: (z == uz[i]) for i in range(min(3, len(uz)))}
    nt = len(truth); Tset = set(truth.tolist()); rel = np.zeros(n); rel[truth] = 1
    rows = []
    for m in METH:
        Ss = []; Ws = []
        for ci in range(4):
            S, w = sampler(m, 1000 * seed + ci); Ss.append(S); Ws.append(w)
        # per-chain mean(가중), 전체 pooled
        chmeans = []
        for ci in range(4):
            wc = Ws[ci]; mc = (wc[:, None] * Ss[ci]).sum(0) / wc.sum() if m == "AWSGLD" else Ss[ci].mean(0)
            chmeans.append(mc)
        allS = torch.cat(Ss, 0); allw = torch.cat(Ws, 0)
        if m == "AWSGLD":
            wn = allw / allw.sum(); mh = (wn[:, None] * allS).sum(0); sd = torch.median(torch.sqrt((wn[:, None] * (allS - mh) ** 2).sum(0)))
        else:
            mh = allS.mean(0); sd = torch.median(allS.std(0))
        mh_c = mh.cpu().numpy(); sd = float(sd)
        # R̂ (split, per-chain thinned)
        segs = [s for c in Ss for s in torch.chunk(c, 2, 0)]
        mm = torch.stack([s.mean(0) for s in segs]); vv = torch.stack([s.var(0, unbiased=True) for s in segs]); L = min(s.shape[0] for s in segs)
        R = torch.median(torch.sqrt(((L - 1) / L * vv.mean(0) + mm.var(0, unbiased=True)) / torch.clamp(vv.mean(0), min=1e-12)))
        # cov90 (θ*)
        lo = torch.quantile(allS, 0.05, 0).cpu().numpy(); hi = torch.quantile(allS, 0.95, 0).cpu().numpy()
        cov90 = float(np.mean((ts >= lo) & (ts <= hi)))
        # Jaccard (4연쇄 top-k)
        chsets = [set(np.argsort(c.cpu().numpy())[::-1][:nt].tolist()) for c in chmeans]
        jp = [len(a & b) / len(a | b) for a, b in itertools.combinations(chsets, 2)]; jac = float(np.mean(jp)) if jp else 1.0
        # 순위·FDR (θ*)
        pih = sigm(mh_c); order = np.argsort(-pih)
        def prf(sel):
            tp = len(set(sel.tolist()) & Tset); Pp = tp / max(len(sel), 1); Rr = tp / max(nt, 1); return Pp, Rr, (2 * Pp * Rr / (Pp + Rr) if Pp + Rr > 0 else 0)
        fdr = {}
        for gmm in GAMMAS:
            k = 1
            for kk in range(1, n + 1):
                if np.mean(1 - pih[order[:kk]]) > gmm: k = kk - 1; break
                k = kk
            sel = order[:max(k, 1)]; Pp, Rr, Fv = prf(sel); fdr[gmm] = (Pp, Rr, Fv, len(set(sel.tolist()) - Tset) / max(len(sel), 1), len(sel))
        pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
        for i in order:
            if i in Tset: tp += 1
            else: fp += 1
            tpr.append(tp / pos); fpr.append(fp / neg)
        auc = float(np.trapezoid(tpr, fpr))
        topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
        disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = (rel[order[:nt]] * disc).sum() / disc.sum()
        mse = float(np.mean((mh_c - ts) ** 2)); sp = spearmanr(mh_c, ts).correlation; kd = kendalltau(mh_c, ts).correlation
        thg = [float(mh_c[grp[gg]].mean()) for gg in range(3)]; pig = [float(pih[grp[gg]].mean()) for gg in range(3)]
        base = [seed, m, float(R), thg[0], thg[1], thg[2], pig[0], pig[1], pig[2],
                sp, kd, topk, ndcg, mse, auc, sd, cov90, jac]
        for gmm in GAMMAS: base += list(fdr[gmm])
        rows.append(base)
        del Ss, Ws, allS; torch.cuda.empty_cache()
    del B, BtB, P, Lc; torch.cuda.empty_cache()
    return rows, nt


def main():
    hdr = ["seed", "method", "rmed", "thS", "thW", "thN", "piS", "piW", "piN",
           "spearman", "kendall", "topk", "ndcg", "mse", "auc", "sigma_raw", "cov90", "jaccard"]
    for gmm in GAMMAS: hdr += [f"P{gmm}", f"R{gmm}", f"F{gmm}", f"rFDR{gmm}", f"nsel{gmm}"]
    rows = []; nt = 0
    for s in range(NSEED):
        r, nt = run_seed(s); rows += r; print(f"[seed {s}] done (n={N}, truth={nt})", flush=True)
    fn = os.path.join(_HERE, (f"redesign_gpu_n{N}_mu{MUGAP}_a{ALPHA}_z{ZETA}" + (f"_k{KDOC}" if KDOC>1 else "") + ".csv"))
    with open(fn, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["#truth", nt, "n", N, "mu", MUGAP, "alpha", ALPHA, "zeta", ZETA]); w.writerow(hdr); w.writerows(rows)
    print(f"저장: {os.path.basename(fn)} ({len(rows)}행)")


if __name__ == "__main__":
    main()
