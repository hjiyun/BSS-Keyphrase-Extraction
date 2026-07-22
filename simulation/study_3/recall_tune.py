"""Study 3 — recall 지향 튜닝: 세 basin(깊은 1994 포함) 고른 방문 → recall↑.
지렛대: TAU(장벽 낮춤) × ZETA(평탄화) × T(탐색). eps0=12 고정.
측정: 3 basin 방문율, R@20/P@20, doc1994 도달율, 발산.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, awsgld, summarize
from keyphrase_functions_awsgld import inv_logit

TB.BASELINE = -2.0
DOCS = ['1994', '212', '227']; SIG2 = 0.5; EPS0 = 12.0; START = 1; NSEED = 5
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = 3; P, L = make_precond(mg['BtB'], n)
ini = mg['centers'][START].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth)

def R_at(pi, k=20):
    sel = np.argsort(-pi)[:k]; return np.isin(sel, truth).sum() / nt
def P_at(pi, k=20):
    sel = np.argsort(-pi)[:k]; return np.isin(sel, truth).sum() / k

print(f"recall 튜닝 (eps0={EPS0}, 출발 doc212, {NSEED}시드): 3basin방문 / R@20 / P@20 / 1994도달 / 발산")
print(f"{'TAU':>4} {'ZETA':>5} {'T':>6} | {'f1994':>6} {'f212':>6} {'f227':>6} | {'R@20':>10} {'P@20':>10} {'1994율':>6} {'발산':>5}")
best = None
for TAU in (1.0, 2.0, 3.0):
    for ZETA in (5.0, 10.0, 20.0):
        for T in (12000,):
            BURN = T // 4
            U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
            fr = np.zeros(K); Rs = []; Ps = []; hit1994 = 0; div = 0
            for s in range(NSEED):
                r = awsgld(U, gU, mode, ini, 1000 + s, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
                mo = r['mode'][BURN:]; f = np.array([(mo == k).mean() for k in range(K)])
                fr += f; hit1994 += (f[0] > 0.02)
                pi = inv_logit(r['theta'][BURN:]).mean(0); Rs.append(R_at(pi)); Ps.append(P_at(pi))
                div += (np.abs(r['theta'][BURN:]).max() > 600)
            fr /= NSEED
            Rm = np.mean(Rs); Pm = np.mean(Ps)
            print(f"{TAU:>4.1f} {ZETA:>5.1f} {T:>6} | {fr[0]:>6.3f} {fr[1]:>6.3f} {fr[2]:>6.3f} | "
                  f"{Rm:>10.3f} {Pm:>10.3f} {hit1994/NSEED:>6.2f} {div:>5}")
            if div == 0 and (best is None or Rm > best[0]):
                best = (Rm, Pm, TAU, ZETA, T)
print(f"\n→ recall 최대(발산X): R@20={best[0]:.3f} P@20={best[1]:.3f}  TAU={best[2]} ZETA={best[3]} T={best[4]}")
