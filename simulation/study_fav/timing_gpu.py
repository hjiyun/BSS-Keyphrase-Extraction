"""알고리즘별 런타임(wall-clock) 측정 — 논문용. 샘플링(T=1)·최적화(T→0) 각 방법 시드1개 시간.

출력: mode / n / iters / method / time_s  (CUDA 동기화 후 측정).
사용: python3 timing_gpu.py <n> [T_samp=5000] [budget_opt=2000]
"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa
DEV = torch.device("cuda"); DT = torch.float64; torch.set_default_dtype(DT)
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
T_S = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
BUD = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
MU, ALPHA, AL, S2, SIGd, PIN, POUT = 1.8, 0.2, 0.5, 1.0, 0.4, 0.30, 0.02
METH = ["tMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def build():
    rng = np.random.default_rng(0); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    ts = np.array([MU, 0.0, -MU])[z] + SIGd * rng.standard_normal(N); A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; n = keep.sum(); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    B = torch.tensor(B, device=DEV); u_0 = torch.tensor(u0, device=DEV); Y = torch.tensor(Y, device=DEV)
    BtB = B.T @ B; ridge = 1e-6 * torch.trace(BtB) / n
    P = torch.linalg.solve(BtB + ridge * torch.eye(n, device=DEV), torch.eye(n, device=DEV)); P = 0.5 * (P + P.T)
    Lc = torch.linalg.cholesky(P + 1e-10 * torch.eye(n, device=DEV))
    def sig(x): return torch.clamp(torch.sigmoid(x), 1e-10, 1 - 1e-10)
    def gradU(th):
        pi = sig(th); tp = torch.clamp((1 - AL) * pi, 1e-10, 1 - 1e-10); dn = torch.clamp(1 - tp, min=1e-10)
        return -torch.where(Y == 1, 1 - pi, -(1 - AL) * pi * (1 - pi) / dn) + (BtB @ (th - u_0)) / S2
    def U(th):
        pi = sig(th); tp = torch.clamp((1 - AL) * pi, 1e-10, 1 - 1e-10)
        ll = (Y * torch.log(tp) + (1 - Y) * torch.log(1 - tp)).sum(); Bd = B @ (th - u_0)
        return (-ll + (Bd @ Bd) / (2 * S2)).item()
    return int(n), P, Lc, gradU, U


def loop(method, n, P, Lc, gradU, U, T, anneal):
    g = torch.Generator(device=DEV); g.manual_seed(0)
    theta = torch.randn(n, generator=g, device=DEV) * 1.5; v = torch.zeros(n, device=DEV)
    aw = torch.arange(1, M_REG + 1, device=DEV, dtype=DT) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        Tt = max(1e-4, np.exp(-t / 400)) if anneal else 1.0
        if method == "tMH":
            prop = theta + 0.25 * np.sqrt(Tt) * (Lc @ torch.randn(n, generator=g, device=DEV)); dU = U(prop) - U(theta)
            if np.log(float(torch.rand((), generator=g, device=DEV)) + 1e-300) < -dU / Tt: theta = prop
        else:
            gr = gradU(theta)
            if method == "SGLD": ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gr + np.sqrt(2 * ek * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "qSGLD": ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ gr) + np.sqrt(2 * ek * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
            elif method == "SGHMC": eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * gr + np.sqrt(2 * 0.1 * eta * Tt) * torch.randn(n, generator=g, device=DEV)
            elif method == "cycSGLD":
                cl = 500; be = (t % cl) / cl; ek = 0.005 * (np.cos(np.pi * min(be, 0.8)) + 1); theta = theta - ek * gr + np.sqrt(2 * Tt * ek) * torch.randn(n, generator=g, device=DEV)
            elif method == "AWSGLD":
                Uv = U(theta); gm = 1.0
                if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
                if t < warm:
                    es.append(Uv)
                    if t == warm - 1: lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
                else: J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1)); gm = float(np.clip(1 + (ZETA / du) * (np.log(float(aw[J]) + 1e-12) - np.log(float(aw[J - 1]) + 1e-12)), 0.1, 10.0))
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ gr) + np.sqrt(2 * ek * gm * Tt) * (Lc @ torch.randn(n, generator=g, device=DEV))
                if t >= warm: dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = float(aw[J]); aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = torch.clamp(aw, 1e-10, 1)
        theta = torch.clamp(theta, -700, 700)


def main():
    n, P, Lc, gradU, U = build()
    print(f"# n={n} (요청 N={N})  샘플링 T={T_S}  최적화 budget={BUD}  [시드1개 wall-clock]")
    print(f"{'mode':>10} {'iters':>6} {'method':>8} {'time_s':>8}")
    for mode, T, anneal in [("sampling", T_S, False), ("optimization", BUD, True)]:
        for m in METH:
            torch.cuda.synchronize(); t0 = time.time()
            loop(m, n, P, Lc, gradU, U, T, anneal)
            torch.cuda.synchronize(); dt = time.time() - t0
            print(f"{mode:>10} {T:>6} {m:>8} {dt:>8.2f}", flush=True)


if __name__ == "__main__":
    main()
