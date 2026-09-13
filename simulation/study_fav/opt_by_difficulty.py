"""난이도별 최적화 수렴 비교 (예산 = 각 난이도의 AWSGLD Tε에 맞춤).

- 같은 데이터에서 최적화 프레임(T→0 annealing)으로 6개 알고리즘 비교.
- 예산: 각 난이도에서 AWSGLD가 ε-optimality에 도달하는 반복수(Tε)의 중앙값 → 그 예산을 모두에 동일 적용.
- 지표(모두, 예산 시점 기준): min U / opt.gap(U-U*) / ‖θ-θ*‖, + 각자 Tε.
- 기준 U*·θ*는 경사하강 최적화로 산출(gold-free).

사용: python3 opt_by_difficulty.py [n_seed=5] [n=100]
난이도: Easy(μ2.5,α0.1) / Moderate(μ1.8,α0.2) / Difficult(μ1.0,α0.4)
"""
import sys
import numpy as np
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0
AL, S2 = 0.5, 1.0; SIGd, PIN, POUT = 0.4, 0.30, 0.02
MAXB = 3000; EPS = 1.0
METH = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
DIFF = [("Easy", 2.5, 0.1), ("Moderate", 1.8, 0.2), ("Difficult", 1.0, 0.4)]
NSEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N     = int(sys.argv[2]) if len(sys.argv) > 2 else 100


def setup(seed, MU, ALPHA):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIGd * rng.standard_normal(N)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u_0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sig(ts)).astype(float)
    return n, B, u_0, Y, B.T @ B


def make(n, B, u_0, Y, BtB):
    def gradU(th):
        pi = np.clip(sig(th), 1e-10, 1 - 1e-10); tp = np.clip((1 - AL) * pi, 1e-10, 1 - 1e-10); dn = np.clip(1 - tp, 1e-10, None)
        return -np.where(Y == 1, 1 - pi, -(1 - AL) * pi * (1 - pi) / dn) + (BtB @ (th - u_0)) / S2
    def U(th):
        pi = np.clip(sig(th), 1e-10, 1 - 1e-10); tp = np.clip((1 - AL) * pi, 1e-10, 1 - 1e-10)
        return -(Y * np.log(tp) + (1 - Y) * np.log(1 - tp)).sum() + ((B @ (th - u_0)) @ (B @ (th - u_0))) / (2 * S2)
    return gradU, U


def run(method, seed, MU, ALPHA, budget):
    """budget 반복까지: 최고(min U)의 U·θ, 그리고 자체 Tε(gap<EPS 첫 반복)."""
    n, B, u_0, Y, BtB = setup(seed, MU, ALPHA); gradU, U = make(n, B, u_0, Y, BtB)
    P = np.linalg.inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    ths = u_0.copy()
    for _ in range(10000):
        ths = ths - 0.05 * gradU(ths)
    Umin = U(ths)
    rng = np.random.default_rng(0); theta = rng.standard_normal(n) * 1.5; v = np.zeros(n)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = np.inf; bth = None; reach = None
    for t in range(budget):
        Tt = max(1e-4, np.exp(-t / 400))
        if method == "acMH":
            prop = theta + 0.25 * np.sqrt(Tt) * (Lc @ rng.standard_normal(n)); dU = U(prop) - U(theta)
            if np.log(rng.random()) < -dU / Tt:
                theta = prop
        else:
            g = gradU(theta)
            if method == "SGLD":
                ek = 0.02 / ((t + 1) ** 0.6 + 10); theta = theta - ek * g + np.sqrt(2 * ek * Tt) * rng.standard_normal(n)
            elif method == "qSGLD":
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * (P @ g) + np.sqrt(2 * ek * Tt) * (Lc @ rng.standard_normal(n))
            elif method == "SGHMC":
                eta = 0.01 / ((t + 1) ** 0.6 + 10); theta = theta + v; v = 0.9 * v - eta * g + np.sqrt(2 * 0.1 * eta * Tt) * rng.standard_normal(n)
            elif method == "cycSGLD":
                cl = 500; be = (t % cl) / cl; ek = 0.005 * (np.cos(np.pi * min(be, 0.8)) + 1)
                theta = theta - ek * g + np.sqrt(2 * Tt * ek) * rng.standard_normal(n)
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
                    gm = float(np.clip(1 + (ZETA / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
                ek = 0.3 / ((t + 1) ** 0.6 + 10); theta = theta - ek * gm * (P @ g) + np.sqrt(2 * ek * gm * Tt) * (Lc @ rng.standard_normal(n))
                if t >= warm:
                    dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]; aw[J:] += dec * cw * (1 - aw[J:]); aw[:J] -= dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        theta = np.clip(theta, -700, 700)
        u = U(theta)
        if u < bU:
            bU = u; bth = theta.copy()
        if reach is None and u - Umin < EPS:
            reach = t
    return bU, bU - Umin, np.linalg.norm(bth - ths), (reach if reach is not None else np.nan)


def main():
    for name, MU, ALPHA in DIFF:
        # 1) AWSGLD Tε 로 예산 결정
        awT = []
        for s in range(NSEED):
            _, _, _, r = run("AWSGLD", s, MU, ALPHA, MAXB)
            awT.append(r if not np.isnan(r) else MAXB - 1)
        budget = int(np.median(awT))
        # 2) 그 예산 하 6개 비교
        res = {m: [run(m, s, MU, ALPHA, budget) for s in range(NSEED)] for m in METH}
        print(f"\n===== 난이도 {name} (μ±{MU}, α={ALPHA}) — 예산=AWSGLD Tε 중앙값={budget}  ({NSEED}시드) =====")
        print(f"{'method':>8} | {'min U':>13} | {'opt.gap(U-U*)':>15} | {'||θ-θ*||':>13} | {'자체Tε':>12}")
        for m in sorted(METH, key=lambda mm: np.mean([r[1] for r in res[mm]])):
            a = np.array(res[m]); rc = a[:, 3][~np.isnan(a[:, 3])]
            rcs = f"{int(np.median(rc))}({len(rc)}/{NSEED})" if len(rc) > 0 else "미도달"
            print(f"{m:>8} | {a[:,0].mean():>6.2f}±{a[:,0].std():>4.2f} | {a[:,1].mean():>7.2f}±{a[:,1].std():>5.2f} | {a[:,2].mean():>6.2f}±{a[:,2].std():>4.2f} | {rcs:>12}")


if __name__ == "__main__":
    main()
