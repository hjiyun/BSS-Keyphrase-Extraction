"""Study 3 — 6개 샘플러 전체 비교 (3문서 & 10문서 트랩).
샘플러: acMH, AWSGLD, SGLD, qSGLD, cycSGLD, SGHMC.
지표: basin 방문 / 탈출율 / 발산율 / top-k(P@20,R@k,F@k) / ROC AUC / NDCG@20.
설정: 각 방법 트랩별 튜닝값. TAU=1.0 공통.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, sgld_family, summarize
from keyphrase_functions_awsgld import inv_logit

TB.BASELINE = -2.0
SIG2 = 0.5; TAU = 1.0; NSEED = 10
SGLD_LR = dict(SGLD=3.0, qSGLD=20.0, cycSGLD=0.3, SGHMC=1.0)

def run_docset(tag, K, T, BURN, acmh_step, aws_eps0, aws_zeta):
    DOCS = TB.pick_low(K)
    mg = TB.build(DOCS, sigma2=SIG2)
    Uraw, gUraw, mode = TB.energies(mg)
    n = mg['n']; P, L = make_precond(mg['BtB'], n)
    U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
    start = K // 2; ini = mg['centers'][start].copy(); im = mode(ini)
    truth = np.array(mg['truth']); nt = len(truth); Tset = set(truth.tolist())
    kR = 20 if K == 3 else 50
    print(f"\n===== {tag}: {K}문서 n={n} truth={nt} 출발=doc{DOCS[im]} "
          f"T={T} | acMH step={acmh_step} AWSGLD eps0={aws_eps0} ζ={aws_zeta} =====", flush=True)

    def auc(pi):
        order = np.argsort(-pi); pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
        for i in order:
            if i in Tset: tp += 1
            else: fp += 1
            tpr.append(tp/pos); fpr.append(fp/neg)
        return float(np.trapezoid(tpr, fpr))
    def ndcg(pi, k=20):
        o = np.argsort(-pi)[:k]; disc = np.log2(np.arange(2, k+2))
        idcg = (np.ones(min(k, nt))/disc[:min(k, nt)]).sum()
        return float((np.isin(o, truth)/disc).sum()/(idcg+1e-12))
    def tk(pi, k):
        sel = np.argsort(-pi)[:k]; tp = np.isin(sel, truth).sum()
        Pp = tp/k; Rr = tp/nt; return Pp, Rr, (2*Pp*Rr/(Pp+Rr) if Pp+Rr > 0 else 0.0)

    def run_one(name, seed):
        if name == 'acMH':
            r = acmh(U, mode, ini, seed, acmh_step, T, P, L)
        elif name == 'AWSGLD':
            r = awsgld(U, gU, mode, ini, 1000+seed, T, P, L, TAU=TAU, ZETA=aws_zeta, eps0=aws_eps0)
        else:
            r = sgld_family(name, U, gU, mode, ini, seed, T, P, L, TAU=TAU, base_lr=SGLD_LR[name])
        return r

    NAMES = ['acMH', 'AWSGLD', 'qSGLD', 'SGLD', 'cycSGLD', 'SGHMC']
    agg = {nm: [] for nm in NAMES}
    t0 = time.time()
    for s in range(NSEED):
        for nm in NAMES:
            r = run_one(nm, s); sm = summarize(r, K, BURN, im)
            div = np.abs(r['theta'][BURN:]).max() > 600
            pi = inv_logit(r['theta'][BURN:]).mean(0)
            P20 = tk(pi, 20); Rk = tk(pi, kR)
            agg[nm].append(dict(vis=sm['visited'], esc=sm['escape_iter'], div=div,
                                p20=P20[0], rk=Rk[1], fk=Rk[2], auc=auc(pi), nd=ndcg(pi)))
        print(f"  seed {s} {int(time.time()-t0)}s", flush=True)

    ms = lambda v: f"{np.mean(v):.3f}({np.std(v):.3f})"
    print(f"\n## {tag} 결과 ({NSEED}시드) — 발산 시드 포함 평균, 발산율 별도")
    print(f"{'샘플러':>8} | {'방문':>10} {'탈출':>6} {'발산':>6} | "
          f"{'P@20':>13} {'R@'+str(kR):>13} {'F@'+str(kR):>13} {'AUC':>13} {'NDCG@20':>13}")
    print("-" * 110)
    for nm in NAMES:
        a = agg[nm]
        vis = ms([x['vis'] for x in a]); esc = np.mean([x['esc'] >= 0 for x in a])
        dv = np.mean([x['div'] for x in a])
        print(f"{nm:>8} | {vis:>10} {esc:>6.2f} {dv:>6.2f} | "
              f"{ms([x['p20'] for x in a]):>13} {ms([x['rk'] for x in a]):>13} "
              f"{ms([x['fk'] for x in a]):>13} {ms([x['auc'] for x in a]):>13} "
              f"{ms([x['nd'] for x in a]):>13}")
    return {nm: agg[nm] for nm in NAMES}

r3 = run_docset("3문서", 3, 12000, 3000, 0.3, 12.0, 10.0)
r10 = run_docset("10문서", 10, 20000, 5000, 0.1, 12.0, 20.0)
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result_all.npz"),
         which='3+10', done=True)
print("\n완료. (result_all.npz)")
