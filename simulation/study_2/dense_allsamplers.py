"""dense 단봉 (실데이터 5문서, full-batch) — 6개 샘플러 비교.
dense_paper_eval.py 와 동일 설정(문서·집계·지표)에 SGLD 계열 4종 추가.
샘플러: acMH(componentwise), AWSGLD(gibbs_mh), SGLD, qSGLD, cycSGLD, SGHMC.
  SGLD 계열은 study_1a/1b sgld_only.py 의 표준 규칙·하이퍼파라미터 그대로 이식
  (BSS σ²-explicit 그래디언트 + σ² Gibbs, floor=0.5, full-batch).
집계: pooled(micro) P/R/F/realFDR + ROC AUC. N_SIM=30, k=5.
"""
import os, csv, time, importlib.util
import numpy as np
from numpy.linalg import solve, cholesky
from scipy.stats import invgamma
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("exp", os.path.join(HERE, "acmh_vs_awsgld_4to10.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
M = exp.M; M.TAU = 1.0; M.ZETA = 5.0; M.DECAY_LR = 100.0; M.M_REGIONS = 1000
LEVELS = exp.FDR_LEVELS; T, BURN, N_SIM, KOBS = exp.T, exp.BURN_IN, 30, 5; d = 0.85
DOCS = ["1949", "2007", "2092", "215", "2017"]
GRID = exp.grid; sigmoid = M.inv_logit

# ---- SGLD 계열 하이퍼파라미터 (sgld_only.py 와 동일) ----
SGLD_TAU = 1.0; SGLD_LR = 0.02; QSGLD_LR = 0.3
CYC_LR = 0.01; CYC_CYCLES = 10
SGHMC_LR = 0.01; SGHMC_FR = 0.1; SGHMC_TAU = 1.0
FLOOR = 0.5

def grad_U(Y, alpha, theta, u_0, sigma2, BtB):     # sgld_only.grad_posterior_energy_fixed_btb (full-batch)
    pi = np.clip(sigmoid(theta), 1e-10, 1 - 1e-10); dpi = pi * (1 - pi)
    temp = np.clip((1 - alpha) * pi, 1e-10, 1 - 1e-10); denom = np.clip(1 - temp, 1e-10, None)
    gll = np.zeros_like(theta); sm = Y == 1
    gll[sm] = 1 - pi[sm]
    gll[~sm] = -(1 - alpha) * dpi[~sm] / denom[~sm]
    gprior = -BtB @ (theta - u_0) / sigma2
    return -(gll + gprior)

def precond(B, n):
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    L = cholesky(P + 1e-10 * np.eye(n))
    return BtB, P, L

def run_sgld(method, n, B, u_0, Y, ini, seed):
    np.random.seed(seed); theta = ini.copy(); alpha = M.alpha_find(theta, Y, GRID)
    BtB, P, L = precond(B, n); ths = np.zeros((T, n)); v = np.zeros(n)
    for t in range(T):
        Bv = B @ (theta - u_0); C = Bv @ Bv
        sigma2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        g = grad_U(Y, alpha, theta, u_0, sigma2, BtB)
        if method == "SGLD":
            eps = SGLD_LR / ((t + 1) ** 0.6 + 10.0)
            theta = theta - eps * g + np.sqrt(2 * SGLD_TAU * eps) * np.random.randn(n)
        elif method == "qSGLD":
            eps = QSGLD_LR / ((t + 1) ** 0.6 + 10.0)
            theta = theta - eps * (P @ g) + np.sqrt(2 * SGLD_TAU * eps) * (L @ np.random.randn(n))
        elif method == "cycSGLD":
            cl = max(1, T // CYC_CYCLES); beta = (t % cl) / cl
            eps = CYC_LR / 2.0 * (np.cos(np.pi * min(beta, 0.8)) + 1.0)
            tau = SGLD_TAU if beta >= 0.8 else SGLD_TAU / 1e4
            theta = theta - eps * g + np.sqrt(2 * tau * eps) * np.random.randn(n)
        elif method == "SGHMC":
            eta = SGHMC_LR / ((t + 1) ** 0.6 + 10.0)
            theta = theta + v
            v = (1 - SGHMC_FR) * v - eta * g + np.sqrt(2 * SGHMC_FR * eta * SGHMC_TAU) * np.random.randn(n)
        theta = np.clip(theta, -700, 700); ths[t] = theta
        alpha = M.alpha_find(theta, Y, GRID)
    return np.mean(sigmoid(ths[BURN:T]), axis=0)

def setup(graph, seed):
    n, truth = graph['n'], graph['truth']
    rng = np.random.RandomState(seed); G = solve(graph['D'], graph['A']); B = np.eye(n) - d * G.T
    w = np.diag(1 / np.sqrt(np.diag(graph['D']))); Bs = np.eye(n) - d * w @ graph['A'] @ w
    Y = np.zeros(n); Y[list(rng.choice(truth, KOBS, replace=False))] = 1
    u0 = solve(B, np.ones(n) * (1 - d)); ini = M.base_to_start(solve(Bs, Y))
    return B, Y, u0, ini, M.alpha_find(u0, Y, GRID)

def roc_auc(pi, tY):
    pos = int(tY.sum()); neg = len(tY) - pos
    if pos == 0 or neg == 0: return np.nan
    order = np.argsort(-pi); tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if tY[i] == 1: tp += 1
        else: fp += 1
        tpr.append(tp/pos); fpr.append(fp/neg)
    return float(np.trapezoid(tpr, fpr))

def topk_prf(pi, truth, k=20):
    sel = np.argsort(-pi)[:k]; tp = np.isin(sel, truth).sum()
    P = tp / k; R = tp / len(truth); return P, R, (2*P*R/(P+R) if P+R > 0 else 0.0)
def ndcg_at(pi, truth, k=20):
    o = np.argsort(-pi)[:k]; disc = np.log2(np.arange(2, k+2)); rel = np.isin(o, truth).astype(float)
    idcg = (np.ones(min(k, len(truth))) / disc[:min(k, len(truth))]).sum()
    return float((rel/disc).sum() / (idcg + 1e-12))

NAMES = ["acMH", "AWSGLD", "qSGLD", "SGLD", "cycSGLD", "SGHMC"]
pool = {m: {g: {'tp': 0., 'pos': 0., 'key': 0.} for g in LEVELS} for m in NAMES}
aucs = {m: [] for m in NAMES}
tk20 = {m: {'P': [], 'R': [], 'F': []} for m in NAMES}
nd20 = {m: [] for m in NAMES}
exp.run_doc(exp.build_graph(DOCS[0]), 1); t0 = time.time()
for doc in DOCS:
    g = exp.build_graph(doc); n, truth = g['n'], g['truth']
    tY = np.zeros(n, int); tY[truth] = 1
    for s in range(N_SIM):
        seed = hash((doc, s)) % (2**31); B, Y, u0, ini, ae = setup(g, seed)
        pis = {}
        cm = exp.componentwise_mcmc_numba(T, ini, n, GRID, ae, u0, B, Y, seed=seed+200000)
        pis['acMH'] = np.mean(sigmoid(cm['theta'])[BURN:T], axis=0)
        np.random.seed(seed+100000)
        pis['AWSGLD'] = M.gibbs_mh(BURN, T, ini, n, g, Y, B, u0, ae, GRID, verbose=False)['poster_pi_mn']
        for m in ["qSGLD", "SGLD", "cycSGLD", "SGHMC"]:
            pis[m] = run_sgld(m, n, B, u0, Y, ini, seed + 300000)
        for m in NAMES:
            ev = exp.eval_metrics(pis[m], Y, truth=truth)
            for gl in LEVELS:
                pool[m][gl]['tp'] += ev[gl]['tp']; pool[m][gl]['pos'] += ev[gl]['pos']; pool[m][gl]['key'] += len(truth)
            aucs[m].append(roc_auc(pis[m], tY))
            P20, R20, F20 = topk_prf(pis[m], truth, 20)
            tk20[m]['P'].append(P20); tk20[m]['R'].append(R20); tk20[m]['F'].append(F20)
            nd20[m].append(ndcg_at(pis[m], truth, 20))
    print(f"[{doc}] n={n} kw={len(truth)} | {int(time.time()-t0)}s", flush=True)

def pooled(m, g):
    tp, pos, key = pool[m][g]['tp'], pool[m][g]['pos'], pool[m][g]['key']
    P = tp/pos if pos > 0 else 0.; R = tp/key if key > 0 else 0.
    return P, R, (2*P*R/(P+R) if P+R > 0 else 0.), ((pos-tp)/pos if pos > 0 else 0.)

print(f"\n=== dense 단봉 5문서 (N_SIM={N_SIM}) — 통일 지표 (P@20/R@20/F@20/ROC AUC/NDCG@20) ===")
print(f"{'샘플러':>8} | {'P@20':>7} {'R@20':>7} {'F@20':>7} {'ROC AUC':>8} {'NDCG@20':>8}")
import csv as _csv
with open("dense_allsamplers_summary.csv", "w", newline="") as _f:
    _w = _csv.writer(_f); _w.writerow(["sampler", "P@20", "R@20", "F@20", "roc_auc", "ndcg20"])
    for m in NAMES:
        p, r, fF = np.mean(tk20[m]['P']), np.mean(tk20[m]['R']), np.mean(tk20[m]['F'])
        au, nd = np.nanmean(aucs[m]), np.mean(nd20[m])
        print(f"{m:>8} | {p:>7.3f} {r:>7.3f} {fF:>7.3f} {au:>8.4f} {nd:>8.3f}")
        _w.writerow([m, round(p, 4), round(r, 4), round(fF, 4), round(au, 4), round(nd, 4)])
print("저장: dense_allsamplers_summary.csv")

print("\n=== pooled F-measure (γ별) ===")
print(f"{'γ':>5} | " + " ".join(f"{m:>8}" for m in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{pooled(m,g)[2]:>8.3f}" for m in NAMES))

print("\n=== pooled Precision (γ별) ===")
print(f"{'γ':>5} | " + " ".join(f"{m:>8}" for m in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{pooled(m,g)[0]:>8.3f}" for m in NAMES))

print("\n=== pooled Recall (γ별) ===")
print(f"{'γ':>5} | " + " ".join(f"{m:>8}" for m in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{pooled(m,g)[1]:>8.3f}" for m in NAMES))

print("\n=== 실현 FDR (γ별) ===")
print(f"{'γ':>5} | " + " ".join(f"{m:>8}" for m in NAMES))
for g in LEVELS:
    print(f"{g:>5} | " + " ".join(f"{pooled(m,g)[3]:>8.3f}" for m in NAMES))

with open(os.path.join(HERE, "dense_allsamplers.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["metric", "gamma", "sampler", "value"])
    for m in NAMES: w.writerow(["roc_auc", "-", m, round(float(np.nanmean(aucs[m])), 4)])
    for g in LEVELS:
        for m in NAMES:
            P, R, F, D = pooled(m, g)
            for nm, val in [("precision", P), ("recall", R), ("F", F), ("realFDR", D)]:
                w.writerow([nm, g, m, round(val, 4)])
print("\n저장: dense_allsamplers.csv")
