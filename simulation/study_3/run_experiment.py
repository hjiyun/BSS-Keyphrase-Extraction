"""Study 3 본 실험 — 실데이터 trap 위 acMH vs AWSGLD (다중 시드).

지형: Hulth 3문서(1994/212/227), 중심=문서별 TextRank, σ²=0.5, baseline=-2.0
      장벽: 1994↔{212,227}=약30(깊음), 212↔227=약8(얕음, 1B 레짐)
설정(튜닝 결과): TAU=1.0, acMH step=0.3, AWSGLD eps0=8 ZETA=5, T=8000 BURN=2000
출발: 가운데 basin(doc 212) — 얕은 탈출(→227) 가능, 깊은 탈출(→1994) 시험
평가: 모드 방문수·탈출·전환 / union-truth top-k(P/R/F) / 문서별 recall
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, summarize
from keyphrase_functions_awsgld import inv_logit

TB.BASELINE = -2.0
SIG2 = 0.5; DOCS = ['1994', '212', '227']
TAU = 1.0; STEP = 0.3; EPS0 = 12.0; ZETA = 10.0
T = 12000; BURN = 3000; NSEED = 10; START = 1        # doc 212 basin

mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = len(DOCS); P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
ini = mg['centers'][START].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth)
truth_doc = mg['truth_doc']
print(f"Study3 본실험: n={n} 문서={DOCS} 출발=doc{DOCS[im]} truth={nt}개 "
      f"| TAU={TAU} acMH_step={STEP} AWSGLD eps0={EPS0} ζ={ZETA} | {NSEED}시드\n", flush=True)

def pimean(res):
    return inv_logit(res['theta'][BURN:]).mean(0)

def topk(pi, k):
    sel = np.argsort(-pi)[:k]; tp = np.isin(sel, truth).sum()
    P = tp / k; R = tp / nt; return P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)

def per_doc_recall(pi, k):
    sel = set(np.argsort(-pi)[:k].tolist())
    return {DOCS[j]: len(sel & set(truth_doc[DOCS[j]])) / max(len(truth_doc[DOCS[j]]), 1)
            for j in range(K)}

rows = {'acMH': [], 'AWSGLD': []}
t0 = time.time()
for s in range(NSEED):
    ra = acmh(U, mode, ini, s, STEP, T, P, L)
    rw = awsgld(U, gU, mode, ini, 1000 + s, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
    for nm, r in (('acMH', ra), ('AWSGLD', rw)):
        sm = summarize(r, K, BURN, im)
        pi = pimean(r); Pk, Rk, Fk = topk(pi, nt)
        rows[nm].append(dict(visited=sm['visited'], escape=sm['escape_iter'],
                             switches=sm['switches'], frac=sm['frac'],
                             P=Pk, R=Rk, F=Fk, pdr=per_doc_recall(pi, nt)))
    print(f"  seed {s} done {int(time.time()-t0)}s | "
          f"acMH방문{rows['acMH'][-1]['visited']} AWS방문{rows['AWSGLD'][-1]['visited']}", flush=True)

def agg(nm, key):
    v = np.array([r[key] for r in rows[nm]], float); return v.mean(), v.std()

print("\n## Study 3 — acMH vs AWSGLD (실데이터 trap, %d시드 평균(표준편차))" % NSEED)
print(f"{'샘플러':>8} | {'방문모드수':>8} {'탈출율':>7} {'전환수':>8} | "
      f"{'top-k P':>9} {'R':>9} {'F':>9}")
print("-" * 74)
for nm in ('acMH', 'AWSGLD'):
    vm, vs = agg(nm, 'visited')
    esc = np.array([r['escape'] for r in rows[nm]]); escrate = (esc >= 0).mean()
    sm_, ss_ = agg(nm, 'switches')
    Pm, Ps = agg(nm, 'P'); Rm, Rs = agg(nm, 'R'); Fm, Fs = agg(nm, 'F')
    print(f"{nm:>8} | {vm:>4.1f}({vs:.1f}) {escrate:>7.2f} {sm_:>5.0f}({ss_:>3.0f}) | "
          f"{Pm:.3f}({Ps:.3f}) {Rm:.3f}({Rs:.3f}) {Fm:.3f}({Fs:.3f})")

print("\n## 모드별 체류 비율 (평균) — 출발=doc %s" % DOCS[im])
print(f"{'샘플러':>8} | " + " ".join(f"doc {d:>5}" for d in DOCS))
for nm in ('acMH', 'AWSGLD'):
    fr = np.mean([r['frac'] for r in rows[nm]], axis=0)
    print(f"{nm:>8} | " + " ".join(f"{f:>9.3f}" for f in fr))

print("\n## 문서별 recall@k (평균) — 각 문서 키워드를 얼마나 회수했나")
print(f"{'샘플러':>8} | " + " ".join(f"doc {d:>5}" for d in DOCS))
for nm in ('acMH', 'AWSGLD'):
    pdr = {d: np.mean([r['pdr'][d] for r in rows[nm]]) for d in DOCS}
    print(f"{nm:>8} | " + " ".join(f"{pdr[d]:>9.3f}" for d in DOCS))

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_result.npz"),
         acMH_frac=np.array([r['frac'] for r in rows['acMH']]),
         AWSGLD_frac=np.array([r['frac'] for r in rows['AWSGLD']]),
         acMH_visited=[r['visited'] for r in rows['acMH']],
         AWSGLD_visited=[r['visited'] for r in rows['AWSGLD']],
         config=dict(TAU=TAU, STEP=STEP, EPS0=EPS0, ZETA=ZETA, start=DOCS[im]))
print("\n저장: experiment_result.npz")
