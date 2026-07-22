"""Study 3 — acMH vs AWSGLD 성능지표 전체 (실데이터 trap).
지표: top-k(P/R/F) at k∈{10,20,49}, ROC AUC, NDCG@20, γ별 FDR-cutoff(P/R/F/실현FDR).
설정은 run_experiment 와 동일 (TAU=1, acMH step=0.3, AWSGLD eps0=12 ζ=5, 10시드).
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld
from keyphrase_functions_awsgld import inv_logit, FDR_cutoff_full

TB.BASELINE = -2.0
SIG2 = 0.5; DOCS = ['1994', '212', '227']
TAU = 1.0; STEP = 0.3; EPS0 = 12.0; ZETA = 10.0
T = 12000; BURN = 3000; NSEED = 10; START = 1
LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
TOPKS = [10, 20]

mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = len(DOCS); P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
ini = mg['centers'][START].copy()
truth = np.array(mg['truth']); nt = len(truth); Tset = set(truth.tolist()); Y = mg['Y']
print(f"Study3 성능지표: n={n} truth={nt} 관측Y={int(Y.sum())} 출발=doc{DOCS[START]} | {NSEED}시드\n", flush=True)

def pimean(res): return inv_logit(res['theta'][BURN:]).mean(0)
def topk(pi, k):
    sel = np.argsort(-pi)[:k]; tp = np.isin(sel, truth).sum()
    Pp = tp / k; Rr = tp / nt; return Pp, Rr, (2*Pp*Rr/(Pp+Rr) if Pp+Rr > 0 else 0.0)
def auc_roc(pi):
    order = np.argsort(-pi); pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr, fpr))
def ndcg(pi, k=20):
    order = np.argsort(-pi)[:k]; disc = np.log2(np.arange(2, k+2))
    rel = np.isin(order, truth).astype(float)
    idcg = (np.ones(min(k, nt)) / disc[:min(k, nt)]).sum()
    return float((rel/disc).sum() / (idcg + 1e-12))
def fdr_row(pi):
    out = {}
    for g in LEVELS:
        pos, tp, rfdr = FDR_cutoff_full(pi.copy(), g, Y, truth)
        Pp = tp/pos if pos > 0 else 0.0; Rr = tp/nt
        out[g] = (Pp, Rr, 2*Pp*Rr/(Pp+Rr) if Pp+Rr > 0 else 0.0, rfdr, pos)
    return out

M = {'acMH': [], 'AWSGLD': []}
t0 = time.time()
for s in range(NSEED):
    ra = acmh(U, mode, ini, s, STEP, T, P, L)
    rw = awsgld(U, gU, mode, ini, 1000+s, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
    for nm, r in (('acMH', ra), ('AWSGLD', rw)):
        pi = pimean(r)
        rec = dict(auc=auc_roc(pi), ndcg=ndcg(pi, 20), fdr=fdr_row(pi))
        for k in TOPKS: rec[f'top{k}'] = topk(pi, k)
        M[nm].append(rec)
    print(f"  seed {s} done {int(time.time()-t0)}s", flush=True)

def ms(vals): return f"{np.mean(vals):.3f}({np.std(vals):.3f})"

print("\n## 표 1 — top-k / ROC AUC / NDCG@20 (10시드 평균(표준편차))")
hdr = f"{'샘플러':>8} |"
for k in TOPKS: hdr += f" {'P@'+str(k):>12} {'R@'+str(k):>12} {'F@'+str(k):>12} |"
hdr += f" {'ROC AUC':>12} {'NDCG@20':>12}"
print(hdr); print("-" * len(hdr))
for nm in ('acMH', 'AWSGLD'):
    row = f"{nm:>8} |"
    for k in TOPKS:
        Ps = [r[f'top{k}'][0] for r in M[nm]]; Rs = [r[f'top{k}'][1] for r in M[nm]]; Fs = [r[f'top{k}'][2] for r in M[nm]]
        row += f" {ms(Ps):>12} {ms(Rs):>12} {ms(Fs):>12} |"
    row += f" {ms([r['auc'] for r in M[nm]]):>12} {ms([r['ndcg'] for r in M[nm]]):>12}"
    print(row)

print("\n## 표 2 — γ별 FDR-cutoff (P / R / F / 실현FDR / 선택수)")
print(f"{'γ':>5} {'샘플러':>8} | {'Precision':>12} {'Recall':>12} {'F':>12} {'실현FDR':>12} {'선택수':>7}")
print("-" * 76)
for g in LEVELS:
    for nm in ('acMH', 'AWSGLD'):
        rows = [r['fdr'][g] for r in M[nm]]
        Pp = [x[0] for x in rows]; Rr = [x[1] for x in rows]; Ff = [x[2] for x in rows]
        rf = [x[3] for x in rows]; ps = [x[4] for x in rows]
        print(f"{g:>5} {nm:>8} | {ms(Pp):>12} {ms(Rr):>12} {ms(Ff):>12} {ms(rf):>12} {np.mean(ps):>7.1f}")
    print("-" * 76)
print("\n저장 없음(표만).")
