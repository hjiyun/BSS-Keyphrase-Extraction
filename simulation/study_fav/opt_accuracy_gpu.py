"""annealing(T→0) 최적화 결과의 best θ 에서 최적화 지표 + 정확도 지표 동시 산출.

- 최적화: min U / opt.gap(U−U*) / ‖θ−θ*‖
- 정확도(annealing 점추정 π=sigmoid(θ*)):
    precision/recall/F1 (일반 분류, π>0.5 기준, FDR컷 아님)
    Top-k(k=truth수) / NDCG@k / MSE(θ vs ts) / AUC
합성데이터라 참 ts·truth 있음. α·σ² 고정(nuisance), θ만 최적화.

사용: python3 opt_accuracy_gpu.py <nseed> <n> <mu> <alpha>
"""
import os, sys, time
import numpy as np
import torch
from scipy.stats import spearmanr
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
MU    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
AL_FIX, S2 = None, 1.0; SIG, PIN, POUT = 0.4, 0.30, 0.02; MAXB = 3000; EPS = 1.0
METH = ["tMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIG * rng.standard_normal(N)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
    truth = np.where(ts > 0)[0]
    return n, B, u0, ts, Y, a0, truth


def build(seed):
    n, Bn, u0n, ts, Yn, a0, truth = gen(seed)
    B = torch.tensor(Bn, device=DEV); u_0 = torch.tensor(u0n, device=DEV); Y = torch.tensor(Yn, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
    alpha = torch.tensor(a0, device=DEV)
    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def gradU(th):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        gll = torch.where(Y == 1, 1 - pi, -(1 - alpha) * pi * (1 - pi) / dn)
        return -gll + (BtB @ (th - u_0)) / S2
    def U(th):
        pi = sig(th); tp = torch.clamp((1 - alpha) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum(); Bd = B @ (th - u_0)
        return -ll + (Bd @ Bd) / (2 * S2)
    return n, u_0, P, Lc, gradU, U, ts, truth


def run(method, seed, budget, pack):
    n, u_0, P, Lc, gradU, U, ts, truth = pack
    ths = u_0.clone()
    for _ in range(8000): ths = ths - 0.05 * gradU(ths)
    Umin = float(U(ths))
    g = torch.Generator(device=DEV); g.manual_seed(0)
    theta = torch.randn(n, generator=g, device=DEV) * 1.5; v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = float("inf"); bth = None
    torch.cuda.synchronize(); t_start = time.time()
    for t in range(budget):
        Tt = max(1e-4, np.exp(-t / 400))
        if method == "tMH":
            prop = theta + 0.25 * np.sqrt(Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            dU = float(U(prop)) - float(U(theta))
            if np.log(float(torch.rand((), generator=g, device=DEV)) + 1e-300) < -dU / Tt: theta = prop
        else:
            gr = gradU(theta)
            if method == "SGLD":
                ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gr + np.sqrt(2 * ek * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "qSGLD":
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gr) + np.sqrt(2 * ek * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            elif method == "SGHMC":
                eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gr + np.sqrt(2 * 0.1 * eta * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "cycSGLD":
                cl = 500; be = (t % cl) / cl; ek = 0.005 * (np.cos(np.pi * min(be, 0.8)) + 1)
                theta = theta - ek * gr + np.sqrt(2 * Tt * ek) * torch.randn(n, generator=g, device=DEV)
            elif method == "AWSGLD":
                Uv = float(U(theta)); gm = 1.0
                if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
                if t < warm:
                    es.append(Uv)
                    if t == warm - 1:
                        lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
                else:
                    J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1))
                    gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ gr) + np.sqrt(2 * ek * gm * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
                if t >= warm:
                    dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
        theta = torch.clamp(theta, -700, 700)
        u = float(U(theta))
        if u < bU: bU = u; bth = theta.clone()
    torch.cuda.synchronize(); dt = time.time() - t_start
    dist = float(torch.linalg.norm(bth - ths))
    # π 공간 gap (최종 관심사 = 확률)
    pistar = sigm(ths.cpu().numpy())
    # 정확도 (best θ 점추정)
    th = bth.cpu().numpy(); pi = sigm(th)
    pidist = float(np.linalg.norm(pi - pistar))          # ‖π−π*‖
    pirms = float(np.sqrt(np.mean((pi - pistar) ** 2)))  # 노드당 평균 확률오차(RMS)
    nt = len(truth); Tset = set(truth.tolist()); n_ = n
    pred = np.where(pi > 0.5)[0]                      # 일반 분류(π>0.5)
    tp = len(set(pred.tolist()) & Tset)
    prec = tp / max(len(pred), 1); rec = tp / max(nt, 1); f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    order = np.argsort(-pi)
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
    rel = np.zeros(n_); rel[truth] = 1; disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = float((rel[order[:nt]] * disc).sum() / disc.sum())
    mse = float(np.mean((th - ts) ** 2))
    spear = float(spearmanr(th, ts).correlation)
    pos = nt; neg = n_ - nt; tpc = fpc = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tpc += 1
        else: fpc += 1
        tpr.append(tpc / pos); fpr.append(fpc / max(neg, 1))
    auc = float(np.trapezoid(tpr, fpr))
    return dict(minU=bU, gap=bU - Umin, dist=dist, pidist=pidist, pirms=pirms, spear=spear, topk=topk, ndcg=ndcg, mse=mse, auc=auc, time_s=dt)


def main():
    packs = [build(s) for s in range(NSEED)]
    awT = []
    for s in range(NSEED):
        # AWSGLD Tε 로 예산
        pk = packs[s]; n, u_0, P, Lc, gradU, U, ts, truth = pk
        ths = u_0.clone()
        for _ in range(8000): ths = ths - 0.05 * gradU(ths)
        Umin = float(U(ths))
        g = torch.Generator(device=DEV); g.manual_seed(0); theta = torch.randn(n, generator=g, device=DEV) * 1.5
        aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1; reach = None
        for t in range(MAXB):
            Tt = max(1e-4, np.exp(-t / 400)); gr = gradU(theta)
            Uv = float(U(theta)); gm = 1.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
            ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = torch.clamp(theta - ek * gm * (P @ gr) + np.sqrt(2 * ek * gm * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV)), -700, 700)
            if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
            if reach is None and float(U(theta)) - Umin < EPS: reach = t
        awT.append(reach if reach is not None else MAXB - 1)
    budget = int(np.median(awT))
    res = {m: [run(m, s, budget, packs[s]) for s in range(NSEED)] for m in METH}

    def ms(m, k, f="{:.3f}"):
        v = [res[m][s][k] for s in range(NSEED)]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    nt = len(packs[0][7])
    print(f"\nannealing 최적화 점추정 지표  n={N} μ±{MU} α={ALPHA}  예산={budget}  ({NSEED}시드, truth≈{nt})")
    print("[최적화]  min U / opt.gap / ‖θ−θ*‖(로짓) / ‖π−π*‖(확률) / π-RMS(노드당) / 시간(초/시드)")
    print(f"  {'method':>8} | {'min U':>15} | {'opt.gap':>13} | {'||θ-θ*||':>12} | {'||π-π*||':>12} | {'π-RMS':>12} | {'time_s':>12}")
    for m in sorted(METH, key=lambda mm: np.mean([res[mm][s]['gap'] for s in range(NSEED)])):
        print(f"  {m:>8} | {ms(m,'minU','{:.2f}'):>15} | {ms(m,'gap','{:.2f}'):>13} | {ms(m,'dist','{:.2f}'):>12} | {ms(m,'pidist','{:.2f}'):>12} | {ms(m,'pirms','{:.4f}'):>12} | {ms(m,'time_s','{:.2f}'):>12}")
    print("\n[정확도(π=sigmoid(θ*) 점추정)]  Spearman / Top-k / NDCG / MSE / AUC")
    print(f"  {'method':>8} | {'Spearman':>12} {'Top-k':>12} {'NDCG':>12} {'MSE':>10} {'AUC':>12}")
    for m in METH:
        print(f"  {m:>8} | {ms(m,'spear'):>12} {ms(m,'topk'):>12} {ms(m,'ndcg'):>12} {ms(m,'mse','{:.2f}'):>10} {ms(m,'auc'):>12}")


if __name__ == "__main__":
    main()
