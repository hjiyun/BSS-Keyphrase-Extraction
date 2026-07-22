"""Study 3 (10문서) — 재튜닝: acMH step / AWSGLD eps0×ζ.
목표: 같은 온도(TAU)에서 acMH 는 갇히고 AWSGLD 는 여러 basin 방문.
측정: 방문 basin 수, 탈출율, R@20, P@20, 발산.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, summarize
from keyphrase_functions_awsgld import inv_logit

TB.BASELINE = -2.0
SIG2 = 0.5; K = 10; NSEED = 5; T = 20000; BURN = 5000; TAU = 1.0
DOCS = TB.pick_low(K)
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
START = K // 2                                  # 가운데 basin 에서 출발
ini = mg['centers'][START].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth)
print(f"10문서 재튜닝: n={n} truth={nt} 출발=doc{DOCS[im]} TAU={TAU} T={T} {NSEED}시드\n", flush=True)

def R_at(pi, k=20): return np.isin(np.argsort(-pi)[:k], truth).sum() / nt
def P_at(pi, k=20): return np.isin(np.argsort(-pi)[:k], truth).sum() / k

t0 = time.time()
print("[1] acMH step — 수용률 & 갇힘 확인")
print(f"   {'step':>6} {'수용률':>7} {'방문':>5} {'탈출율':>7} {'R@20':>7} {'P@20':>7}")
acm = {}
for s in (0.1, 0.3, 0.6):
    vis = []; esc = 0; Rs = []; Ps = []
    for sd in range(NSEED):
        r = acmh(U, mode, ini, sd, s, T, P, L); sm = summarize(r, K, BURN, im)
        vis.append(sm['visited']); esc += (sm['escape_iter'] >= 0)
        pi = inv_logit(r['theta'][BURN:]).mean(0); Rs.append(R_at(pi)); Ps.append(P_at(pi))
    acm[s] = (np.mean(vis), esc / NSEED, np.mean(Rs), np.mean(Ps))
    print(f"   {s:>6.2f} {r['accept']:>7.3f} {np.mean(vis):>5.1f} {esc/NSEED:>7.2f} "
          f"{np.mean(Rs):>7.3f} {np.mean(Ps):>7.3f}   ({int(time.time()-t0)}s)", flush=True)
STEP = min(acm, key=lambda s: (acm[s][0], -acm[s][3]))     # 가장 갇히는 step
print(f"   → acMH step={STEP} (방문 {acm[STEP][0]:.1f})\n", flush=True)

print("[2] AWSGLD eps0 × ζ")
print(f"   {'eps0':>6} {'ZETA':>5} | {'방문':>5} {'탈출율':>7} {'R@20':>7} {'P@20':>7} {'경계':>6} {'발산':>5}")
best = None
for eps0 in (12.0, 20.0, 30.0):
    for ZETA in (5.0, 10.0, 20.0):
        vis = []; esc = 0; Rs = []; Ps = []; div = 0; bd = []
        for sd in range(NSEED):
            r = awsgld(U, gU, mode, ini, 1000 + sd, T, P, L, TAU=TAU, ZETA=ZETA, eps0=eps0)
            sm = summarize(r, K, BURN, im); vis.append(sm['visited']); esc += (sm['escape_iter'] >= 0)
            pi = inv_logit(r['theta'][BURN:]).mean(0); Rs.append(R_at(pi)); Ps.append(P_at(pi))
            div += (np.abs(r['theta'][BURN:]).max() > 600); bd.append(r['boundary_rate'])
        Rm, Pm, vm = np.mean(Rs), np.mean(Ps), np.mean(vis)
        print(f"   {eps0:>6.1f} {ZETA:>5.1f} | {vm:>5.1f} {esc/NSEED:>7.2f} {Rm:>7.3f} {Pm:>7.3f} "
              f"{np.mean(bd):>6.2f} {div:>5}   ({int(time.time()-t0)}s)", flush=True)
        if div == 0 and (best is None or Rm > best[0]):
            best = (Rm, Pm, vm, eps0, ZETA)
print(f"\n→ 선택: acMH step={STEP} | AWSGLD eps0={best[3]} ζ={best[4]} "
      f"(방문 {best[2]:.1f}, R@20={best[0]:.3f}, P@20={best[1]:.3f})")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned10.npz"),
         step=STEP, eps0=best[3], zeta=best[4], TAU=TAU, docs=np.array(DOCS), T=T, BURN=BURN)
print("저장: tuned10.npz")
