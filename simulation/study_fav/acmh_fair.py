"""공정 재비교 — acMH(논문 원본) vs SG-MCMC 5종, 둘 다 정식 프로토콜.

- SG-MCMC(SGLD·qSGLD·cycSGLD·SGHMC·AWSGLD): redesign_gpu 와 동일한 정식 프로토콜
  = α·σ² Langevin 샘플링(σ² marginal 타깃과 정합) + 4체인 pooling + thinning. GPU.
- acMH: 논문 componentwise_mcmc 원본(σ² 적분·α 적응). CPU. 단일 체인.
- 같은 데이터(합성 μ±1.2,α0.35, 참 ts·truth), n=1000, T=3000, 3시드.
지표: Spearman/Top-k/NDCG/AUC(정확도) · σ̂/cov90(불확실성) · time(초/시드, 순수 루프)

사용: python3 acmh_fair.py [n=1000] [nseed=3] [T=3000]
"""
import os, sys, time, itertools
import numpy as np
import torch
from numpy.linalg import solve
from scipy.stats import spearmanr
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
N     = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 3
T     = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
BURN  = T // 2; THIN = 5
MU, ALPHA, SIG, PIN, POUT = 1.2, 0.35, 0.4, 0.30, 0.02
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0
SG = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIG * rng.standard_normal(N)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * solve(np.diag(deg), A).T; u0 = solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
    truth = np.where(ts > 0)[0]
    return n, B, u0, Y, a0, ts, truth


def sg_sampler(method, seed_ci, n, B, u_0, Y, BtB, P, Lc, a0, g):
    g.manual_seed(seed_ci); theta = torch.randn(n, generator=g, device=DEV) * 1.5
    aa = torch.log(torch.tensor(a0 / (1 - a0), device=DEV)); ss = torch.zeros((), device=DEV); v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    keep = []; wts = []
    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    for t in range(T):
        alpha = torch.sigmoid(aa); s2 = torch.exp(torch.clamp(ss, -6, 6))
        pi = sig(theta); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        gth = -torch.where(Y == 1, 1 - pi, -(1 - alpha) * pi * (1 - pi) / dn) + (BtB @ (theta - u_0)) / s2
        if method == "SGLD":
            ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gth + np.sqrt(2 * ek) * torch.randn(n, generator=g, device=DEV)
        elif method == "qSGLD":
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gth) + np.sqrt(2 * ek) * (Lc @ torch.randn(n, generator=g, device=DEV))
        elif method == "cycSGLD":
            cl = max(1, T // 10); be = (t % cl) / cl; ek = 0.01 / 2 * (np.cos(np.pi * min(be, 0.8)) + 1); tk = 1.0 if be >= 0.8 else 1e-4
            theta = theta - ek * gth + np.sqrt(2 * tk * ek) * torch.randn(n, generator=g, device=DEV)
        elif method == "SGHMC":
            eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gth + np.sqrt(2 * 0.1 * eta) * torch.randn(n, generator=g, device=DEV)
        elif method == "AWSGLD":
            Uv = float((-(Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum() + ((B @ (theta - u_0)) @ (B @ (theta - u_0))) / (2 * s2)))
            gm = 1.0
            if not np.isfinite(Uv): Uv = float(emin) if emin is not None else 0.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ gth) + np.sqrt(2 * ek * gm) * (Lc @ torch.randn(n, generator=g, device=DEV))
            if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
        theta = torch.clamp(theta, -700, 700)
        # α, σ² Langevin (재매개변수화)
        pi = sig(theta); tt = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        gA = ((Y / (1 - alpha) - (1 - Y) * pi / (1 - tt)).sum() - 1 / alpha + 1 / (1 - alpha)) * alpha * (1 - alpha)
        C = ((B @ (theta - u_0)) ** 2).sum(); gS = -(C / 2 + 0.001) / s2 + (n / 2 + 0.001)
        eA = 0.2 * (100.0 / n) / ((t + 1) ** 0.6 + 10); eS = eA
        aa = torch.clamp(aa - eA * gA + np.sqrt(2 * eA) * torch.randn((), generator=g, device=DEV), -6, 6)
        ss = torch.clamp(ss - eS * gS + np.sqrt(2 * eS) * torch.randn((), generator=g, device=DEV), -6, 6)
        if t >= BURN and t % THIN == 0: keep.append(theta.clone()); wts.append(float(aw[J]) if method == "AWSGLD" else 1.0)
    return torch.stack(keep), torch.tensor(wts, device=DEV)


def optref(Bn, u0n, Yn, a0, n):
    """GD 로 MAP θ* (α=a0, σ²=1 고정) → U*, π*."""
    BtB = Bn.T @ Bn; th = u0n.copy()
    def gradU(t):
        pi = np.clip(sigm(t), 1e-10, 1 - 1e-10); tp = np.clip((1 - a0) * pi, 1e-10, 1 - 1e-10); dn = np.clip(1 - tp, 1e-10, None)
        return -np.where(Yn == 1, 1 - pi, -(1 - a0) * pi * (1 - pi) / dn) + (BtB @ (t - u0n))
    for _ in range(8000): th = th - 0.05 * gradU(th)
    return float(Ubatch(th[None], Bn, u0n, Yn, a0)[0]), sigm(th)


def Ubatch(S, Bn, u0n, Yn, a0):
    """샘플 배치 S(m,n) 의 U(θ) (α=a0, σ²=1). 반환 (m,)."""
    pi = np.clip(sigm(S), 1e-10, 1 - 1e-10); tp = np.clip((1 - a0) * pi, 1e-10, 1 - 1e-10)
    ll = (Yn[None] * np.log(tp) + (1 - Yn)[None] * np.log(1 - tp)).sum(1)
    R = (S - u0n) @ Bn.T; quad = (R ** 2).sum(1)
    return -ll + quad / 2


def sg_metrics(method, seed, n, B, u_0, Y, BtB, P, Lc, a0, ts, truth, Bn, u0n, Yn, ustar, pistar):
    g = torch.Generator(device=DEV)
    torch.cuda.synchronize(); t0 = time.time()
    Ss = []; Ws = []
    for ci in range(4):
        S, w = sg_sampler(method, 1000 * seed + ci, n, B, u_0, Y, BtB, P, Lc, a0, g); Ss.append(S); Ws.append(w)
    torch.cuda.synchronize(); dt = time.time() - t0
    allS = torch.cat(Ss, 0); allw = torch.cat(Ws, 0)
    if method == "AWSGLD":
        wn = allw / allw.sum(); mh = (wn[:, None] * allS).sum(0); sd = torch.median(torch.sqrt((wn[:, None] * (allS - mh) ** 2).sum(0)))
    else:
        mh = allS.mean(0); sd = torch.median(allS.std(0))
    lo = torch.quantile(allS, 0.05, 0).cpu().numpy(); hi = torch.quantile(allS, 0.95, 0).cpu().numpy()
    Snp = allS.cpu().numpy()
    m = _metric(mh.cpu().numpy(), Snp, float(sd), lo, hi, ts, truth, n, Bn, u0n, Yn, a0, ustar, pistar); m['time'] = dt
    del Ss, Ws, allS; torch.cuda.empty_cache()
    return m


def acmh_metrics(seed, n, B, u0, Y, a0, ts, truth, ustar, pistar):
    ini = np.random.default_rng(100 + seed).standard_normal(n) * 1.5
    t0 = time.time(); r = kfa.componentwise_mcmc(T, ini, n, kfa.grid, a0, u0, B, Y, verbose=False); dt = time.time() - t0
    S = r['theta'][BURN:]
    mh = S.mean(0); sd = float(np.median(S.std(0)))
    lo = np.quantile(S, 0.05, 0); hi = np.quantile(S, 0.95, 0)
    m = _metric(mh, S, sd, lo, hi, ts, truth, n, B, u0, Y, a0, ustar, pistar); m['time'] = dt
    return m


def _metric(mh, S, sd, lo, hi, ts, truth, n, Bn, u0n, Yn, a0, ustar, pistar):
    minU = float(Ubatch(S, Bn, u0n, Yn, a0).min()); gap = minU - ustar
    pihat = sigm(mh); pidist = float(np.linalg.norm(pihat - pistar)); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2)))
    cov90 = float(np.mean((ts >= lo) & (ts <= hi)))
    pi = sigm(mh); order = np.argsort(-pi); nt = len(truth); Tset = set(truth.tolist())
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
    rel = np.zeros(n); rel[truth] = 1; disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = float((rel[order[:nt]] * disc).sum() / disc.sum())
    spear = float(spearmanr(mh, ts).correlation)
    pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / max(neg, 1))
    auc = float(np.trapezoid(tpr, fpr))
    return dict(minU=minU, gap=gap, pidist=pidist, pirms=pirms, spear=spear, topk=topk, ndcg=ndcg, auc=auc, sigma=sd, cov90=cov90)


def main():
    METH = ["acMH"] + SG
    R = {m: [] for m in METH}
    for s in range(NSEED):
        n, Bn, u0n, Yn, a0, ts, truth = gen(s)
        B = torch.tensor(Bn, device=DEV); u_0 = torch.tensor(u0n, device=DEV); Y = torch.tensor(Yn, device=DEV)
        BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
        P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
        Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
        ustar, pistar = optref(Bn, u0n, Yn, a0, n)
        for m in SG:
            R[m].append(sg_metrics(m, s, n, B, u_0, Y, BtB, P, Lc, a0, ts, truth, Bn, u0n, Yn, ustar, pistar))
        R["acMH"].append(acmh_metrics(s, n, Bn, u0n, Yn, a0, ts, truth, ustar, pistar))
        print(f"[seed {s}] done (n={n}, truth={len(truth)}, U*={ustar:.1f})", flush=True)
        del B, u_0, Y, BtB, P, Lc; torch.cuda.empty_cache()

    def ms(m, k, f="{:.3f}"):
        v = [R[m][s][k] for s in range(NSEED)]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n공정 재비교 (정식 프로토콜)  n={N} μ±{MU} α={ALPHA}  T={T}(burn {BURN})  {NSEED}시드")
    print(f"  {'method':>8} | {'min U':>13} {'opt.gap':>11} {'||π-π*||':>10} {'π-RMS':>10} | {'Spearman':>11} {'Top-k':>11} {'NDCG':>11} {'AUC':>11} | {'sigma':>9} {'cov90':>11} | {'time(s)':>11}")
    for m in METH:
        print(f"  {m:>8} | {ms(m,'minU','{:.1f}'):>13} {ms(m,'gap','{:.2f}'):>11} {ms(m,'pidist','{:.2f}'):>10} {ms(m,'pirms','{:.4f}'):>10} | {ms(m,'spear'):>11} {ms(m,'topk'):>11} {ms(m,'ndcg'):>11} {ms(m,'auc'):>11} | {ms(m,'sigma','{:.2f}'):>9} {ms(m,'cov90'):>11} | {ms(m,'time','{:.1f}'):>11}")


if __name__ == "__main__":
    main()
