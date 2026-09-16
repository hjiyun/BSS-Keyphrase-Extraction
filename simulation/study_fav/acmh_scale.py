"""원논문 componentwise acMH 확장성(비용) 실측 + 대형 n 투영.

componentwise_mcmc: sweep 1회 = n개 성분 각각 MH, 각 성분마다 log_posterior 2회(‖B(θ-u0)‖² 포함).
→ sweep당 O(n^2)~O(n^3). 소규모에서 sec/sweep 실측 → 스케일 지수 적합 → n=2000/5000/10000 총시간 투영.
(T=5000 sweep 기준: SG-MCMC 는 수 초, acMH 는 시간~일 → '비용상 불가' 근거.)

사용: python3 acmh_scale.py
"""
import os, sys, time
import numpy as np
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa

MU, ALPHA, SIG, PIN, POUT = 1.2, 0.35, 0.4, 0.30, 0.02
NS = [100, 200, 400, 800, 1600]
SWEEPS = 4                     # 실측용 소량 sweep (첫 sweep 워밍업 제외)
T_FULL = 5000                  # 실제 실험이 요구하는 sweep 수
sigm = lambda x: 1 / (1 + np.exp(-np.clip(x, -700, 700)))


def gen(N, seed=0):
    rng = np.random.default_rng(seed); z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    mu = np.array([MU, 0.0, -MU]); ts = mu[z] + SIG * rng.standard_normal(N)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(z == z[i], PIN, POUT); r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; n = len(ts); deg = A.sum(1)
    B = np.eye(n) - 0.85 * np.linalg.solve(np.diag(deg), A).T; u0 = np.linalg.solve(B, np.ones(n) * 0.15)
    Y = (rng.random(n) < (1 - ALPHA) * sigm(ts)).astype(np.float64)
    a0 = kfa.alpha_find(u0, Y, kfa.grid)
    return n, B, u0, Y, float(a0)


def sec_per_sweep(N):
    n, B, u0, Y, a0 = gen(N)
    ini = np.random.default_rng(1).standard_normal(n) * 1.5
    # 워밍업 1 sweep(=T 2) 후 SWEEPS 회 실측
    t0 = time.time()
    kfa.componentwise_mcmc(SWEEPS + 1, ini, n, kfa.grid, a0, u0, B, Y, verbose=False)
    dt = time.time() - t0
    return n, dt / SWEEPS


def main():
    print("componentwise acMH 확장성 실측 (μ±1.2, α0.35)\n")
    print(f"{'n':>6} {'sec/sweep':>11} {'투영 T=5000 총시간':>22}")
    ns = []; sps = []
    for N in NS:
        n, sp = sec_per_sweep(N); ns.append(n); sps.append(sp)
        tot = sp * T_FULL
        print(f"{n:>6} {sp:>11.4f} {str(__import__('datetime').timedelta(seconds=int(tot))):>22}", flush=True)
    # 로그-로그 스케일 지수
    ln = np.log(ns); ls = np.log(sps); p = np.polyfit(ln, ls, 1)
    print(f"\n스케일: sec/sweep ∝ n^{p[0]:.2f}  (이론 componentwise O(n²)~O(n³) 구간)")
    print(f"\n대형 n 투영 (T=5000 sweep 총시간):")
    for N in (2000, 5000, 10000):
        sp = np.exp(np.polyval(p, np.log(N))); tot = sp * T_FULL
        print(f"  n={N:>6}: {sp:.3f} s/sweep → 총 {str(__import__('datetime').timedelta(seconds=int(tot)))}  (SG-MCMC 는 수 초)")


if __name__ == "__main__":
    main()
