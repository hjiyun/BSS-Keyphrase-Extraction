"""Study 3 — 실데이터 trap 위 6 샘플러 전체 성능지표 (acMH/AWSGLD/SGLD/qSGLD/cycSGLD/SGHMC).
run_metrics.py(3문서)·run10_metrics.py(10문서) 를 6종으로 확장 + CSV 저장.
SGLD 계열은 base_lr 을 방문 basin 최대로 1시드 튜닝(공정) 후 본실행.
지표: top-k(P/R/F), ROC AUC, NDCG@20, γ별 FDR-cutoff(P/R/F/실현FDR/선택수).
실행: python3 run_metrics_all.py 3      # 3문서
      python3 run_metrics_all.py 10     # 10문서
"""
import os, sys, csv, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import trap_build as TB
from trap_samplers import make_precond, acmh, awsgld, sgld_family, summarize
from keyphrase_functions_awsgld import inv_logit, FDR_cutoff_full

KDOC = int(sys.argv[1]) if len(sys.argv) > 1 else 3
TB.BASELINE = -2.0; SIG2 = 0.5; TAU = 1.0; NSEED = 10
if KDOC == 3:
    DOCS = ['1994', '212', '227']; STEP = 0.3; EPS0 = 12.0; ZETA = 10.0
    T = 12000; BURN = 3000; START = 1
    LEVELS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]; TOPKS = [10, 20]
else:
    DOCS = TB.pick_low(10); STEP = 0.1; EPS0 = 12.0; ZETA = 20.0
    T = 20000; BURN = 5000; START = 5
    LEVELS = [0.05, 0.10, 0.15, 0.20, 0.30]

mg = TB.build(DOCS, sigma2=SIG2)
Uraw, gUraw, mode = TB.energies(mg)
n = mg['n']; K = len(DOCS); P, L = make_precond(mg['BtB'], n)
U = lambda th: Uraw(th) / TAU; gU = lambda th: gUraw(th) / TAU
START = min(START, K - 1); ini = mg['centers'][START].copy(); im = mode(ini)
truth = np.array(mg['truth']); nt = len(truth); Tset = set(truth.tolist()); Y = mg['Y']
if KDOC != 3: TOPKS = [20, 50, nt]
NAMES = ['acMH', 'AWSGLD', 'SGLD', 'qSGLD', 'cycSGLD', 'SGHMC']
print(f"Study3 {KDOC}문서 6샘플러: n={n} truth={nt} 관측Y={int(Y.sum())} 출발=doc{DOCS[im]} "
      f"| T={T} {NSEED}시드\n", flush=True)

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

# ---- SGLD 계열 base_lr 튜닝 (1시드, 방문 basin 최대) ----
GRID = {"SGLD": [0.5, 2, 8, 30], "qSGLD": [0.2, 0.5, 2, 8],
        "cycSGLD": [0.2, 0.5, 2, 8], "SGHMC": [0.1, 0.5, 2, 8]}
LR = {}
print("[튜닝] SGLD 계열 base_lr (방문 basin 최대)", flush=True)
for m, lrs in GRID.items():
    best = None
    for lr in lrs:
        r = sgld_family(m, U, gU, mode, ini, 0, T, P, L, TAU=TAU, base_lr=float(lr))
        div = np.abs(r['theta'][BURN:]).max() > 600
        vis = summarize(r, K, BURN, im)['visited']
        if not div and (best is None or vis > best[0]): best = (vis, lr)
    LR[m] = best[1] if best else lrs[0]
    print(f"   {m:>8}: lr={LR[m]}", flush=True)
print(flush=True)

def run(nm, seed):
    if nm == 'acMH': return acmh(U, mode, ini, seed, STEP, T, P, L)
    if nm == 'AWSGLD': return awsgld(U, gU, mode, ini, 1000+seed, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
    return sgld_family(nm, U, gU, mode, ini, seed, T, P, L, TAU=TAU, base_lr=LR[nm])

M = {nm: [] for nm in NAMES}
t0 = time.time()
for s in range(NSEED):
    for nm in NAMES:
        r = run(nm, s); pi = pimean(r)
        summ = summarize(r, K, BURN, im)
        rec = dict(auc=auc_roc(pi), ndcg=ndcg(pi, 20), fdr=fdr_row(pi),
                   visited=summ['visited'], escaped=1.0 if summ['escape_iter'] >= 0 else 0.0)
        for k in TOPKS: rec[f'top{k}'] = topk(pi, k)
        M[nm].append(rec)
    print(f"  seed {s} done {int(time.time()-t0)}s", flush=True)

def ms(vals): return f"{np.mean(vals):.3f}({np.std(vals):.3f})"
mean = lambda vals: float(np.mean(vals)); std = lambda vals: float(np.std(vals))

# ---- 통일 요약: Basins / Escape / P@20 / R@20 / F@20 / AUC / NDCG@20 ----
print(f"\n## 통일 요약 (Basins/Escape/P@20/R@20/F@20/AUC/NDCG@20, {NSEED}시드 평균)")
print(f"{'샘플러':>8} {'Basins':>7} {'Escape':>7} {'P@20':>7} {'R@20':>7} {'F@20':>7} {'AUC':>7} {'NDCG@20':>8}")
with open(f"trap{KDOC}_summary_all.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["sampler", "basins", "escape", "P@20", "R@20", "F@20", "auc", "ndcg20"])
    for nm in NAMES:
        bas = mean([r['visited'] for r in M[nm]]); esc = mean([r['escaped'] for r in M[nm]])
        p20 = mean([r['top20'][0] for r in M[nm]]); r20 = mean([r['top20'][1] for r in M[nm]])
        f20 = mean([r['top20'][2] for r in M[nm]])
        au = mean([r['auc'] for r in M[nm]]); nd = mean([r['ndcg'] for r in M[nm]])
        print(f"{nm:>8} {bas:>7.1f} {esc:>7.2f} {p20:>7.3f} {r20:>7.3f} {f20:>7.3f} {au:>7.3f} {nd:>8.3f}")
        w.writerow([nm, round(bas, 2), round(esc, 2), round(p20, 4), round(r20, 4), round(f20, 4), round(au, 4), round(nd, 4)])

print(f"\n## top-k / ROC AUC / NDCG@20 ({NSEED}시드 평균(표준편차))")
hdr = f"{'샘플러':>8} |"
for k in TOPKS: hdr += f" {'P@'+str(k):>11} {'F@'+str(k):>11} |"
hdr += f" {'ROC AUC':>11} {'NDCG@20':>11}"
print(hdr); print("-" * len(hdr))
for nm in NAMES:
    row = f"{nm:>8} |"
    for k in TOPKS:
        Ps = [r[f'top{k}'][0] for r in M[nm]]; Fs = [r[f'top{k}'][2] for r in M[nm]]
        row += f" {ms(Ps):>11} {ms(Fs):>11} |"
    row += f" {ms([r['auc'] for r in M[nm]]):>11} {ms([r['ndcg'] for r in M[nm]]):>11}"
    print(row)

print(f"\n## γ별 FDR-cutoff F-measure ({NSEED}시드 평균)")
print(f"{'γ':>5} | " + " ".join(f"{nm:>8}" for nm in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{mean([r['fdr'][g][2] for r in M[nm]]):>8.3f}" for nm in NAMES))
print(f"\n## γ별 실현 FDR ({NSEED}시드 평균)")
print(f"{'γ':>5} | " + " ".join(f"{nm:>8}" for nm in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{mean([r['fdr'][g][3] for r in M[nm]]):>8.3f}" for nm in NAMES))

# ---- CSV 저장 ----
tag = f"trap{KDOC}"
with open(f"{tag}_topk_auc_all.csv", "w", newline="") as f:
    w = csv.writer(f); cols = ["sampler"]
    for k in TOPKS: cols += [f"P@{k}", f"P@{k}_sd", f"R@{k}", f"R@{k}_sd", f"F@{k}", f"F@{k}_sd"]
    cols += ["auc", "auc_sd", "ndcg20", "ndcg20_sd"]; w.writerow(cols)
    for nm in NAMES:
        row = [nm]
        for k in TOPKS:
            for i in range(3):
                vals = [r[f'top{k}'][i] for r in M[nm]]; row += [round(mean(vals), 4), round(std(vals), 4)]
        row += [round(mean([r['auc'] for r in M[nm]]), 4), round(std([r['auc'] for r in M[nm]]), 4),
                round(mean([r['ndcg'] for r in M[nm]]), 4), round(std([r['ndcg'] for r in M[nm]]), 4)]
        w.writerow(row)
with open(f"{tag}_fdr_by_gamma_all.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["gamma", "sampler", "precision", "precision_sd", "recall", "recall_sd",
                                   "F", "F_sd", "realFDR", "realFDR_sd", "n_selected"])
    for g in LEVELS:
        for nm in NAMES:
            rows = [r['fdr'][g] for r in M[nm]]
            cells = [g, nm]
            for i in range(4):
                vals = [x[i] for x in rows]; cells += [round(mean(vals), 4), round(std(vals), 4)]
            cells += [round(mean([x[4] for x in rows]), 1)]
            w.writerow(cells)
print(f"\n저장: {tag}_topk_auc_all.csv, {tag}_fdr_by_gamma_all.csv")
