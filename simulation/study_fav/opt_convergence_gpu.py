"""최적화 수렴 비교 (GPU판, 대형 n). 예산 = AWSGLD Tε에 맞춤.

- 목표: U(θ) 최소화(MAP/최빈점) = 최적화. 6개 알고리즘 동일 annealing(T→0) 공정 비교.
- torch dense matvec(GPU) → n=5000·10000 가능. MH는 tMH(블록, annealed)로 대체(componentwise acMH는 대형 n 불가).
- 지표: min U / opt.gap(U-U*) / ‖θ-θ*‖ / Tε.  기준 U*·θ*는 GD 최적화(gold-free).
- α·σ² 고정(nuisance), θ만.

사용: python3 opt_convergence_gpu.py <n_seed> <n> <mu> <alpha> [eps=1.0]
"""
import os, sys
import numpy as np
import torch
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MU    = float(sys.argv[3]) if len(sys.argv) > 3 else 1.8
ALPHA = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
KDOC  = int(sys.argv[5]) if len(sys.argv) > 5 else 1
EPS   = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
PCROSS = 0.004
AL, S2 = 0.5, 1.0; SIGd, PIN, POUT = 0.4, 0.30, 0.02; MAXB = 3000
METH = ["tMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIGd * rng.standard_normal(N)
    doc = rng.integers(0, KDOC, N) if KDOC > 1 else np.zeros(N, int)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(doc == doc[i], np.where(z == z[i], PIN, POUT), PCROSS); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u_0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    return n, B, u_0, Y


def build(seed):
    n, Bn, u0n, Yn = gen(seed)
    B = torch.tensor(Bn, device=DEV); u_0 = torch.tensor(u0n, device=DEV); Y = torch.tensor(Yn, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))

    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def gradU(th):
        pi = sig(th); tp = torch.clamp((1 - AL) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        gll = torch.where(Y == 1, 1 - pi, -(1 - AL) * pi * (1 - pi) / dn)
        return -gll + (BtB @ (th - u_0)) / S2
    def U(th):
        pi = sig(th); tp = torch.clamp((1 - AL) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum(); Bd = B @ (th - u_0)
        return (-ll + (Bd @ Bd) / (2 * S2)).item()
    return n, B, u_0, Y, BtB, P, Lc, gradU, U


def run(method, seed, budget, pack=None):
    if pack is None:
        pack = build(seed)
    n, B, u_0, Y, BtB, P, Lc, gradU, U = pack
    ths = u_0.clone()
    for _ in range(8000):
        ths = ths - 0.05 * gradU(ths)
    Umin = U(ths)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    theta = torch.randn(n, generator=g, device=DEV) * 1.5; v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = float("inf"); bth = None; reach = None
    for t in range(budget):
        Tt = float(os.environ['TCONST']) if os.environ.get('TCONST') else max(1e-4, np.exp(-t / 400))
        if method == "tMH":
            prop = theta + 0.25 * np.sqrt(Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            dU = U(prop) - U(theta)
            if np.log(float(torch.rand((), generator=g, device=DEV))) < -dU / Tt:
                theta = prop
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
                Uv = U(theta); gm = 1.0
                if not np.isfinite(Uv):
                    Uv = emin if emin is not None else 0.0
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
        u = U(theta)
        if u < bU:
            bU = u; bth = theta.clone()
        if reach is None and u - Umin < EPS:
            reach = t
    dist = float(torch.linalg.norm(bth - ths))
    return bU, bU - Umin, dist, (reach if reach is not None else np.nan), pack


def main():
    packs = [build(s) for s in range(NSEED)]
    # 1) AWSGLD Tε 로 예산 결정
    awT = []
    for s in range(NSEED):
        _, _, _, r, _ = run("AWSGLD", s, MAXB, packs[s]); awT.append(r if not np.isnan(r) else MAXB - 1)
    budget = int(os.environ['BFIX']) if os.environ.get('BFIX') else int(np.median(awT))
    res = {m: [] for m in METH}
    for m in METH:
        for s in range(NSEED):
            res[m].append(run(m, s, budget, packs[s])[:4])
    print(f"\n최적화(GPU)  n={N} μ±{MU} α={ALPHA}  예산=AWSGLD Tε 중앙값={budget}  ({NSEED}시드)")
    print(f"{'method':>8} | {'min U':>15} | {'opt.gap(U-U*)':>15} | {'||θ-θ*||':>13} | {'자체Tε':>12}")
    for m in sorted(METH, key=lambda mm: np.mean([r[1] for r in res[mm]])):
        a = np.array(res[m]); rc = a[:, 3][~np.isnan(a[:, 3])]
        rcs = f"{int(np.median(rc))}({len(rc)}/{NSEED})" if len(rc) > 0 else "미도달"
        print(f"{m:>8} | {a[:,0].mean():>8.2f}±{a[:,0].std():>5.2f} | {a[:,1].mean():>7.2f}±{a[:,1].std():>5.2f} | {a[:,2].mean():>6.2f}±{a[:,2].std():>4.2f} | {rcs:>12}")


if __name__ == "__main__":
    main()
