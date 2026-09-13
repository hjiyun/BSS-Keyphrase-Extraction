"""최적화 수렴 비교 (고정 예산 하 '누가 U_min=MAP에 얼마나/빨리 도달하나').

- 목표: U(θ) 최소화(MAP/최빈점 θ* 찾기) = 최적화 프레임.
- 6개 알고리즘(acMH·SGLD·qSGLD·cycSGLD·SGHMC·AWSGLD)을 **동일 annealing(온도→0)** 으로 공정 비교.
- 기준 U*·θ*는 경사하강 최적화로 산출(gold-free, 정답 불필요).
- α·σ²는 고정(nuisance), θ만 최적화. AWSGLD는 gm/aw self-adjust 포함.

지표:
  min U   : 예산 내 도달한 최저 에너지값
  opt.gap : U(best) - U*        (0에 가까울수록 최저점 도달)
  ‖θ-θ*‖ : best-θ 와 최적점 거리
  Tε      : U-U* < eps 를 처음 만족한 반복수 (iterations to ε-optimality). 없으면 '미도달'

사용: python3 opt_convergence.py <n_seed> <n> <mu> <alpha> [budget=1500] [eps=1.0]
예:   python3 opt_convergence.py 5 100 1.2 0.35
"""
import os, sys
import numpy as np
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))
M_REG = kfa.M_REGIONS; DECAY = kfa.DECAY_LR; ZETA = 1.0

NSEED  = int(sys.argv[1]) if len(sys.argv) > 1 else 5
N      = int(sys.argv[2]) if len(sys.argv) > 2 else 100
MU     = float(sys.argv[3]) if len(sys.argv) > 3 else 1.2
ALPHA  = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35     # 데이터 생성 라벨잡음
BUDGET = int(sys.argv[5]) if len(sys.argv) > 5 else 1500
EPS    = float(sys.argv[6]) if len(sys.argv) > 6 else 1.0
AL, S2 = 0.5, 1.0                                             # 에너지의 α·σ² 고정(nuisance)
SIG, PIN, POUT = 0.4, 0.30, 0.02
METH = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]


def setup(seed):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIG * rng.standard_normal(N)
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


def run(method, seed):
    n, B, u_0, Y, BtB = setup(seed); gradU, U = make(n, B, u_0, Y, BtB)
    P = np.linalg.inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    ths = u_0.copy()                                          # 기준 최적점 θ* (gold-free)
    for _ in range(10000):
        ths = ths - 0.05 * gradU(ths)
    Umin = U(ths)
    rng = np.random.default_rng(0); theta = rng.standard_normal(n) * 1.5; v = np.zeros(n)
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG; warm = 300; es = []; emin = du = None; J = M_REG - 1
    bU = np.inf; bth = None; reach = None
    for t in range(BUDGET):
        Tt = max(1e-4, np.exp(-t / 400))                      # 동일 annealing(온도→0)
        if method == "acMH":                                  # 시뮬레이티드 어닐링 MH
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
            elif method == "AWSGLD":                          # gm·aw self-adjust
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
    res = {m: [run(m, s) for s in range(NSEED)] for m in METH}
    print(f"\n최적화 수렴 비교  n={N} μ±{MU} α={ALPHA}  예산={BUDGET}  ε={EPS}  ({NSEED}시드 mean±std)")
    print("(동일 annealing exp(-t/400), α·σ² 고정, θ만; 기준 U*·θ*는 최적화로 산출=gold-free)\n")
    print(f"{'method':>8} | {'min U':>13} | {'opt.gap(U-U*)':>15} | {'||θ-θ*||':>13} | {'Tε(iters→ε)':>13}")
    order = sorted(METH, key=lambda m: np.mean([r[1] for r in res[m]]))   # gap 오름차순
    for m in order:
        a = np.array(res[m]); rc = a[:, 3][~np.isnan(a[:, 3])]
        rcs = f"{int(np.median(rc))}({len(rc)}/{NSEED})" if len(rc) > 0 else "미도달"
        print(f"{m:>8} | {a[:,0].mean():>6.2f}±{a[:,0].std():>4.2f} | {a[:,1].mean():>7.2f}±{a[:,1].std():>5.2f} | {a[:,2].mean():>6.2f}±{a[:,2].std():>4.2f} | {rcs:>13}")


if __name__ == "__main__":
    main()
