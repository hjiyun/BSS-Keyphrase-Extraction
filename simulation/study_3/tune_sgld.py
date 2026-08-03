"""Study 3 — SGLD 계열(SGLD/qSGLD/cycSGLD/SGHMC) 학습률 스캔 (3문서 트랩).
목표: 발산 없이 탐색이 일어나는 base_lr. 측정: 방문 basin / R@20 / 최대변위 / 발산.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, sgld_family, summarize
from keyphrase_functions_awsgld import inv_logit

TB.BASELINE = -2.0
DOCS = ['1994', '212', '227']; SIG2 = 0.5; TAU = 1.0; T = 12000; BURN = 3000; NSEED = 3
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = 3; P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
ini = mg['centers'][1].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth)
R = lambda pi, k=20: np.isin(np.argsort(-pi)[:k], truth).sum() / nt
print(f"SGLD 계열 LR 스캔 (3문서, 출발 doc212, {NSEED}시드): 방문/R@20/변위/발산\n", flush=True)

GRIDS = {
    "SGLD":    [3, 10, 30, 100, 300],      # 무전처리 → 큰 LR
    "qSGLD":   [3, 8, 12, 20, 40],         # 전처리 (AWSGLD ζ=0 유사)
    "cycSGLD": [3, 10, 30, 100, 300],      # 무전처리
    "SGHMC":   [0.3, 1, 3, 10, 30],        # 운동량 → 작은 eta
}
t0 = time.time()
best = {}
for method, lrs in GRIDS.items():
    print(f"[{method}]")
    for lr in lrs:
        vis = []; Rs = []; disp = []; div = 0
        for s in range(NSEED):
            r = sgld_family(method, U, gU, mode, ini, s, T, P, L, TAU=TAU, base_lr=float(lr))
            sm = summarize(r, K, BURN, im); vis.append(sm['visited'])
            pi = inv_logit(r['theta'][BURN:]).mean(0); Rs.append(R(pi))
            d = np.linalg.norm(r['theta'][BURN:] - ini, axis=1).max(); disp.append(d)
            div += (np.abs(r['theta'][BURN:]).max() > 600)
        print(f"   lr={lr:>5} 방문={np.mean(vis):.1f} R@20={np.mean(Rs):.3f} "
              f"변위={np.mean(disp):.1f} 발산={div}/{NSEED}  ({int(time.time()-t0)}s)", flush=True)
        if div == 0 and (method not in best or np.mean(Rs) > best[method][0]):
            best[method] = (np.mean(Rs), lr, np.mean(vis))
    print()
print("→ 선택 (발산X, R@20 최대):")
for m, (Rm, lr, vm) in best.items():
    print(f"   {m}: base_lr={lr}  (방문 {vm:.1f}, R@20 {Rm:.3f})")
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tuned_sgld.npz"),
         **{m: best[m][1] for m in best})
print("저장: tuned_sgld.npz")
