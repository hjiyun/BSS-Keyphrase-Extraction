"""Study 3 (10문서) — acMH vs AWSGLD 전 지표.
설정: TAU=1.0 | acMH step=0.1 (수용률 0.178; step>=0.3 은 수용률 0 = 동결이라 제외)
      AWSGLD eps0=12, ζ=20 | T=20000, BURN=5000, 10시드, 출발=가운데 basin
지표: basin 방문/탈출, top-k(k=20/50/156), ROC AUC, NDCG@20, γ별 FDR-cutoff, 문서별 recall
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, summarize
from keyphrase_functions_awsgld import inv_logit, FDR_cutoff_full

TB.BASELINE = -2.0
SIG2 = 0.5; K = 10; TAU = 1.0; STEP = 0.1; EPS0 = 12.0; ZETA = 20.0
T = 20000; BURN = 5000; NSEED = 10
LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]
DOCS = TB.pick_low(K)
mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
START = K // 2
ini = mg['centers'][START].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth); Tset = set(truth.tolist()); Y = mg['Y']
truth_doc = mg['truth_doc']
TOPKS = [20, 50, nt]
print(f"10문서 전지표: n={n} truth={nt} 관측Y={int(Y.sum())} 출발=doc{DOCS[im]} "
      f"| TAU={TAU} step={STEP} eps0={EPS0} ζ={ZETA} | {NSEED}시드\n", flush=True)

def topk(pi, k):
    sel = np.argsort(-pi)[:k]; tp = np.isin(sel, truth).sum()
    Pp = tp/k; Rr = tp/nt; return Pp, Rr, (2*Pp*Rr/(Pp+Rr) if Pp+Rr > 0 else 0.0)
def auc_roc(pi):
    order = np.argsort(-pi); pos = nt; neg = n-nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr, fpr))
def ndcg(pi, k=20):
    order = np.argsort(-pi)[:k]; disc = np.log2(np.arange(2, k+2))
    rel = np.isin(order, truth).astype(float)
    idcg = (np.ones(min(k, nt))/disc[:min(k, nt)]).sum()
    return float((rel/disc).sum()/(idcg+1e-12))
def pdr(pi, k):
    sel = set(np.argsort(-pi)[:k].tolist())
    return {d: len(sel & set(truth_doc[d]))/max(len(truth_doc[d]), 1) for d in DOCS}

M = {'acMH': [], 'AWSGLD': []}
t0 = time.time()
for s in range(NSEED):
    ra = acmh(U, mode, ini, s, STEP, T, P, L)
    rw = awsgld(U, gU, mode, ini, 1000+s, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
    for nm, r in (('acMH', ra), ('AWSGLD', rw)):
        sm = summarize(r, K, BURN, im); pi = inv_logit(r['theta'][BURN:]).mean(0)
        rec = dict(visited=sm['visited'], escape=sm['escape_iter'], frac=sm['frac'],
                   auc=auc_roc(pi), ndcg=ndcg(pi), pdr=pdr(pi, nt),
                   fdr={g: FDR_cutoff_full(pi.copy(), g, Y, truth) for g in LEVELS})
        for k in TOPKS: rec[f'top{k}'] = topk(pi, k)
        M[nm].append(rec)
    print(f"  seed {s} done {int(time.time()-t0)}s | acMH방문{M['acMH'][-1]['visited']} "
          f"AWS방문{M['AWSGLD'][-1]['visited']}", flush=True)

ms = lambda v: f"{np.mean(v):.3f}({np.std(v):.3f})"
print("\n## ① 탐색 — basin 방문 / 탈출 (10 basin 중)")
print(f"{'샘플러':>8} | {'방문 basin':>12} {'탈출율':>7}")
for nm in ('acMH', 'AWSGLD'):
    esc = np.array([r['escape'] for r in M[nm]])
    print(f"{nm:>8} | {ms([r['visited'] for r in M[nm]]):>12} {(esc >= 0).mean():>7.2f}")

print("\n## ② 순위 — top-k / AUC / NDCG   (R@20 이론상한 = 20/%d = %.3f)" % (nt, 20/nt))
hdr = f"{'샘플러':>8} |"
for k in TOPKS: hdr += f" {'P@'+str(k):>12} {'R@'+str(k):>12} {'F@'+str(k):>12} |"
hdr += f" {'ROC AUC':>12} {'NDCG@20':>12}"
print(hdr); print("-" * len(hdr))
for nm in ('acMH', 'AWSGLD'):
    row = f"{nm:>8} |"
    for k in TOPKS:
        row += (f" {ms([r[f'top{k}'][0] for r in M[nm]]):>12}"
                f" {ms([r[f'top{k}'][1] for r in M[nm]]):>12}"
                f" {ms([r[f'top{k}'][2] for r in M[nm]]):>12} |")
    row += f" {ms([r['auc'] for r in M[nm]]):>12} {ms([r['ndcg'] for r in M[nm]]):>12}"
    print(row)

print("\n## ③ FDR-cutoff (γ별 P / R / F / 실현FDR / 선택수)")
print(f"{'γ':>5} {'샘플러':>8} | {'Precision':>12} {'Recall':>12} {'F':>12} {'실현FDR':>12} {'선택수':>7}")
for g in LEVELS:
    print("-" * 76)
    for nm in ('acMH', 'AWSGLD'):
        rr = [r['fdr'][g] for r in M[nm]]
        pos = [x[0] for x in rr]; tp = [x[1] for x in rr]; rf = [x[2] for x in rr]
        Pp = [t/p if p > 0 else 0.0 for t, p in zip(tp, pos)]; Rr = [t/nt for t in tp]
        Ff = [2*a*b/(a+b) if a+b > 0 else 0.0 for a, b in zip(Pp, Rr)]
        print(f"{g:>5} {nm:>8} | {ms(Pp):>12} {ms(Rr):>12} {ms(Ff):>12} {ms(rf):>12} {np.mean(pos):>7.1f}")

print("\n## ④ 문서별 recall@%d (10문서 각각의 키워드 회수율)" % nt)
print(f"{'샘플러':>8} | " + " ".join(f"{d:>6}" for d in DOCS))
for nm in ('acMH', 'AWSGLD'):
    v = {d: np.mean([r['pdr'][d] for r in M[nm]]) for d in DOCS}
    print(f"{nm:>8} | " + " ".join(f"{v[d]:>6.3f}" for d in DOCS))
print(f"{'회수>0.5 문서수':>8} | acMH "
      f"{sum(np.mean([r['pdr'][d] for r in M['acMH']]) > 0.5 for d in DOCS)}개, AWSGLD "
      f"{sum(np.mean([r['pdr'][d] for r in M['AWSGLD']]) > 0.5 for d in DOCS)}개")

print("\n## ⑤ basin 체류 비율 (평균)")
print(f"{'샘플러':>8} | " + " ".join(f"{d:>6}" for d in DOCS))
for nm in ('acMH', 'AWSGLD'):
    fr = np.mean([r['frac'] for r in M[nm]], axis=0)
    print(f"{nm:>8} | " + " ".join(f"{f:>6.3f}" for f in fr))

np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result10.npz"),
         acMH_frac=np.array([r['frac'] for r in M['acMH']]),
         AWSGLD_frac=np.array([r['frac'] for r in M['AWSGLD']]),
         docs=np.array(DOCS), nt=nt, n=n)
print("\n저장: result10.npz")
