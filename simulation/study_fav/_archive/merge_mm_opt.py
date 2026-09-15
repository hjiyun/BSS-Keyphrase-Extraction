"""합성 다봉 — 최적화 모드(annealing T→0)로 min U / opt.gap / ‖θ−θ*‖ 측정.

- U_mix(θ) = -logsumexp_k(-U_k) (다봉). 기준 U*·θ*는 다중출발 L-BFGS로 전역최소.
- 6샘플러를 동일 annealing(온도→0)으로 돌려 전역최소 도달 비교.
  → AWSGLD는 trap 탈출로 전역최소 도달(gap 작음), 갇힌 샘플러는 지역최소(gap 큼).
사용: python3 merge_mm_opt.py <KDOC> [nseed=5] [n=200]
"""
import os, sys, csv
import numpy as np
from numpy.linalg import solve, cholesky
from scipy.special import logsumexp
from scipy.optimize import minimize
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
from keyphrase_functions_awsgld import alpha_find, inv_logit, grid

KDOC  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N     = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MU, SIG, ALPHA = 1.8, 0.4, 0.2
PIN, POUT, PCROSS = 0.30, 0.02, 0.004
SIGMA2, OBS_RATIO = 0.5, 0.20
BUDGET, EPS = 8000, 1.0
M_REG, DECAY, ZETA = 1000, 100.0, 10.0
NAMES = ['acMH', 'SGLD', 'qSGLD', 'cycSGLD', 'SGHMC', 'AWSGLD']
d = 0.85


def textrank(A):
    dg = A.sum(1); dg[dg == 0] = 1
    B = np.eye(len(A)) - d * solve(np.diag(dg), A).T
    return B, solve(B, np.ones(len(A)) * (1 - d))


def gen(seed):
    rng = np.random.default_rng(seed)
    z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    ts = np.array([MU, 0.0, -MU])[z] + SIG * rng.standard_normal(N)
    doc = rng.integers(0, KDOC, N) if KDOC > 1 else np.zeros(N, int)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(doc == doc[i], np.where(z == z[i], PIN, POUT), PCROSS)
        r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    keep = A.sum(1) > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; z = z[keep]; doc = doc[keep]; n = len(ts)
    B, u_bar = textrank(A)
    centers = []
    for k in range(KDOC):
        m = np.where(doc == k)[0]; u = np.full(n, -MU)
        if len(m) >= 2 and A[np.ix_(m, m)].sum() > 0:
            _, uk = textrank(A[np.ix_(m, m)]); u[m] = uk
        centers.append(u)
    if KDOC == 1: centers = [u_bar]
    Y = np.zeros(n)
    for k in range(max(KDOC, 1)):
        pos = np.where((doc == k) & (ts > 0))[0] if KDOC > 1 else np.where(ts > 0)[0]
        if len(pos):
            mm = max(1, round(OBS_RATIO * len(pos))); Y[rng.choice(pos, min(mm, len(pos)), replace=False)] = 1.0
    a = alpha_find(np.mean(centers, axis=0), Y, grid)
    return n, B, B.T @ B, centers, Y, a


def energies(B, BtB, centers, Y, a):
    s2 = SIGMA2
    def lik(th):
        t = np.clip((1 - a) * inv_logit(th), 1e-10, 1 - 1e-10)
        return -float(np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t)))
    def glik(th):
        p = np.clip(inv_logit(th), 1e-10, 1 - 1e-10); dp = p * (1 - p)
        t = np.clip((1 - a) * p, 1e-10, 1 - 1e-10)
        return -(Y * (1 - a) * dp / t - (1 - Y) * (1 - a) * dp / (1 - t))
    def Uk(th, u): r = B @ (th - u); return lik(th) + float(r @ r) / (2 * s2)
    def U(th): return -float(logsumexp([-Uk(th, u) for u in centers]))
    def gU(th):
        Us = np.array([-Uk(th, u) for u in centers]); w = np.exp(Us - logsumexp(Us))
        g = np.zeros_like(th)
        for wi, u in zip(w, centers): g += wi * (glik(th) + BtB @ (th - u) / s2)
        return g
    return U, gU


def optimize(method, U, gU, P, Lc, n, seed):
    rng = np.random.RandomState(seed); theta = rng.randn(n) * 1.5; v = np.zeros(n)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = np.inf; bth = None
    for t in range(BUDGET):
        Tt = max(1e-4, np.exp(-t / 400)); g = gU(theta)
        if method == 'acMH':
            prop = theta + 0.3 * np.sqrt(Tt) * (Lc @ rng.randn(n)); dU = U(prop) - U(theta)
            if np.log(rng.rand() + 1e-300) < -dU / Tt: theta = prop
        elif method == 'SGLD':
            ek = 0.1 / ((t + 1) ** 0.6 + 10); theta = theta - ek * g + np.sqrt(2 * ek * Tt) * rng.randn(n)
        elif method == 'qSGLD':
            ek = 2.0 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ g) + np.sqrt(2 * ek * Tt) * (Lc @ rng.randn(n))
        elif method == 'SGHMC':
            eta = 0.05 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * g + np.sqrt(2 * 0.1 * eta * Tt) * rng.randn(n)
        elif method == 'cycSGLD':
            cl = 800; be = (t % cl) / cl; ek = 0.05 * (np.cos(np.pi * min(be, 0.8)) + 1)
            theta = theta - ek * g + np.sqrt(2 * Tt * ek) * rng.randn(n)
        elif method == 'AWSGLD':
            Uv = U(theta); gm = 1.0
            if not np.isfinite(Uv): Uv = emin if emin is not None else 0.0
            if t < warm:
                es.append(Uv)
                if t == warm - 1:
                    lo, hi = min(es), max(es); rg = max(hi - lo, 1.0); emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8)
            else:
                J = int(np.clip((Uv - emin) / du + 1, 1, M_REG - 1))
                gm = float(np.clip(1 + (ZETA / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
            ek = 2.0 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ g) + np.sqrt(2 * ek * gm * Tt) * (Lc @ rng.randn(n))
            if t >= warm:
                dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        theta = np.clip(theta, -700, 700); u = U(theta)
        if u < bU: bU = u; bth = theta.copy()
    return bU, bth


def main():
    res = {nm: {'minU': [], 'gap': [], 'dist': []} for nm in NAMES}
    for s in range(NSEED):
        n, B, BtB, centers, Y, a = gen(s); U, gU = energies(B, BtB, centers, Y, a)
        ridge = 1e-6 * np.trace(BtB) / n; P = solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = cholesky(P + 1e-10 * np.eye(n))
        # 전역 최소 U*·θ* (다중출발)
        rng0 = np.random.RandomState(999 + s); starts = [c.copy() for c in centers]
        ub = np.mean(centers, axis=0)
        for _ in range(4): starts.append(ub + rng0.normal(0, 1.5, n))
        best = None
        for x0 in starts:
            rr = minimize(U, x0, jac=gU, method='L-BFGS-B', options=dict(maxiter=3000))
            if best is None or rr.fun < best[0]: best = (float(rr.fun), rr.x)
        Ustar, thstar = best
        for nm in NAMES:
            bU, bth = optimize(nm, U, gU, P, Lc, n, s)
            res[nm]['minU'].append(bU); res[nm]['gap'].append(bU - Ustar); res[nm]['dist'].append(float(np.linalg.norm(bth - thstar)))
        print(f"  seed {s} done (U*={Ustar:.2f})", flush=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"mm_opt_k{KDOC}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(['method', 'minU', 'minU_sd', 'opt_gap', 'opt_gap_sd', 'dist', 'dist_sd'])
        for nm in NAMES:
            a = res[nm]
            w.writerow([nm, f"{np.mean(a['minU']):.3f}", f"{np.std(a['minU']):.3f}", f"{np.mean(a['gap']):.3f}", f"{np.std(a['gap']):.3f}", f"{np.mean(a['dist']):.3f}", f"{np.std(a['dist']):.3f}"])
    print(f"\n합성 다봉 최적화(annealing) K={KDOC} (n={N}, {NSEED}시드)")
    print(f"{'method':>8} | {'min U':>14} | {'opt.gap(U-U*)':>16} | {'||θ-θ*||':>14}")
    for nm in sorted(NAMES, key=lambda m: np.mean(res[m]['gap'])):
        a = res[nm]
        print(f"{nm:>8} | {np.mean(a['minU']):>7.2f}±{np.std(a['minU']):>5.2f} | {np.mean(a['gap']):>8.2f}±{np.std(a['gap']):>6.2f} | {np.mean(a['dist']):>7.2f}±{np.std(a['dist']):>5.2f}")


if __name__ == "__main__":
    main()
