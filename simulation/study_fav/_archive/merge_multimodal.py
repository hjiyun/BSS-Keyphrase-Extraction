"""합성 다봉(문서 K개 log-sum-exp 혼합) + 진짜 θ* 보유 → 전 지표 계산.

- 각 문서 k: 자체 서브그래프의 TextRank 중심 u^(k) (서로 달라 다봉).
- 에너지: U_mix(θ) = -logsumexp_k(-U_k),  U_k = -loglik + ||B(θ-u^(k))||²/(2σ²)  (CLAUDE.md 규칙)
- 진짜 latent θ*(=ts) 보유 → basins/escape(다봉) + γ-FDR/순위/σ̂/cov90/Jaccard(정답) 모두 계산.
- 샘플러 6종: acMH/AWSGLD/SGLD/qSGLD/cycSGLD/SGHMC (study_3 trap_samplers 재사용).

사용: python3 merge_multimodal.py <KDOC> [nseed=5] [n=200]
"""
import os, sys, csv, itertools
import numpy as np
from numpy.linalg import solve
from scipy.special import logsumexp
from scipy.stats import spearmanr
from scipy.optimize import minimize
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_3")
import keyphrase_functions_awsgld as kfa
from keyphrase_functions_awsgld import alpha_find, inv_logit, grid
from trap_samplers import make_precond, acmh, awsgld, sgld_family, summarize

KDOC  = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N     = int(sys.argv[3]) if len(sys.argv) > 3 else 200
MU, SIG, ALPHA = 1.8, 0.4, 0.2
PIN, POUT, PCROSS = 0.30, 0.02, 0.004
SIGMA2, TAU, ZETA, EPS0, STEP = 0.5, 1.0, 10.0, 12.0, 0.3
OBS_RATIO = 0.20
T, BURN = 12000, 3000
GAMMAS = [0.05, 0.10, 0.15, 0.20, 0.30]
NAMES = ['acMH', 'AWSGLD', 'SGLD', 'qSGLD', 'cycSGLD', 'SGHMC']
LR = {'SGLD': 0.1, 'qSGLD': 0.1, 'cycSGLD': 0.1, 'SGHMC': 0.05}
d = 0.85


def textrank(A):
    dg = A.sum(1); dg[dg == 0] = 1
    B = np.eye(len(A)) - d * solve(np.diag(dg), A).T
    return B, solve(B, np.ones(len(A)) * (1 - d))


def gen(seed):
    rng = np.random.default_rng(seed)
    z = rng.choice([0, 1, 2], N, p=[0.4, 0.3, 0.3])
    ts = np.array([MU, 0.0, -MU])[z] + SIG * rng.standard_normal(N)   # 진짜 θ*
    doc = rng.integers(0, KDOC, N) if KDOC > 1 else np.zeros(N, int)
    # 병합 그래프(문서내 SBM + 문서간 희소)
    A = np.zeros((N, N))
    for i in range(N):
        pr = np.where(doc == doc[i], np.where(z == z[i], PIN, POUT), PCROSS)
        r = rng.random(N); r[:i + 1] = 1.0; A[i, r < pr] = 1.0
    A = np.maximum(A, A.T); np.fill_diagonal(A, 0)
    keep = A.sum(1) > 0
    A = A[np.ix_(keep, keep)]; ts = ts[keep]; z = z[keep]; doc = doc[keep]; n = len(ts)
    B, u_bar = textrank(A)
    # 문서별 중심 u^(k): 문서 k 서브그래프 TextRank를 전체 어휘에 배치(나머지 baseline)
    BASELINE = -MU
    centers = []
    for k in range(KDOC):
        m = np.where(doc == k)[0]
        u = np.full(n, BASELINE)
        if len(m) >= 2:
            Ak = A[np.ix_(m, m)]
            if Ak.sum() > 0:
                _, uk = textrank(Ak); u[m] = uk
        centers.append(u)
    if KDOC == 1: centers = [u_bar]
    Y = np.zeros(n)
    for k in range(max(KDOC, 1)):
        pos = np.where((doc == k) & (ts > 0))[0] if KDOC > 1 else np.where(ts > 0)[0]
        if len(pos):
            m = max(1, round(OBS_RATIO * len(pos)))
            Y[rng.choice(pos, min(m, len(pos)), replace=False)] = 1.0
    a = alpha_find(np.mean(centers, axis=0), Y, grid)
    return dict(n=n, A=A, B=B, BtB=B.T @ B, centers=centers, Y=Y, ts=ts,
                truth=np.where(ts > 0)[0], alpha=a)


def energies(mg):
    Y, B, BtB, a, s2, cs = mg['Y'], mg['B'], mg['BtB'], mg['alpha'], SIGMA2, mg['centers']
    def lik(th):
        t = np.clip((1 - a) * inv_logit(th), 1e-10, 1 - 1e-10)
        return -float(np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t)))
    def glik(th):
        p = np.clip(inv_logit(th), 1e-10, 1 - 1e-10); dp = p * (1 - p)
        t = np.clip((1 - a) * p, 1e-10, 1 - 1e-10)
        return -(Y * (1 - a) * dp / t - (1 - Y) * (1 - a) * dp / (1 - t))
    def Uk(th, u): r = B @ (th - u); return lik(th) + float(r @ r) / (2 * s2)
    def Umix(th): return -float(logsumexp([-Uk(th, u) for u in cs]))
    def gUmix(th):
        Us = np.array([-Uk(th, u) for u in cs]); w = np.exp(Us - logsumexp(Us))
        g = np.zeros_like(th)
        for wi, u in zip(w, cs): g += wi * (glik(th) + BtB @ (th - u) / s2)
        return g
    def mode(th): return int(np.argmin([Uk(th, u) for u in cs]))
    return Umix, gUmix, mode


def fdr_select(pi, g):
    o = np.argsort(-pi); cum = np.cumsum(1 - pi[o]); k = np.arange(1, len(pi) + 1)
    ok = np.where(cum / k <= g)[0]
    S = np.zeros(len(pi), bool)
    if len(ok): S[o[:ok[-1] + 1]] = True
    return S


def auc_roc(pi, truth):
    o = np.argsort(pi); r = np.empty(len(pi)); r[o] = np.arange(1, len(pi) + 1)
    tr = np.zeros(len(pi), bool); tr[truth] = True
    n1 = tr.sum(); n0 = (~tr).sum()
    return (r[tr].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else 0.5


def ndcg(pi, truth, k=20):
    tr = np.zeros(len(pi), bool); tr[truth] = True
    o = np.argsort(-pi)[:k]; gains = tr[o] / np.log2(np.arange(2, k + 2))
    ideal = np.sort(tr)[::-1][:k] / np.log2(np.arange(2, k + 2))
    return gains.sum() / ideal.sum() if ideal.sum() else 0.0


def run():
    acc = {nm: {} for nm in NAMES}
    topk_sets = {nm: [] for nm in NAMES}
    for s in range(NSEED):
        mg = gen(s); n = mg['n']; ts = mg['ts']; truth = mg['truth']
        U, gU, mode = energies(mg); P, L = make_precond(mg['BtB'], n)
        Utau = lambda th: U(th) / TAU; gUtau = lambda th: gU(th) / TAU
        ini = mg['centers'][0].copy(); im = mode(ini)
        # 전역 최소 U*·θ* (다중출발: 각 중심 + 랜덤)
        rng0 = np.random.RandomState(999 + s); starts = [c.copy() for c in mg['centers']]
        ub = np.mean(mg['centers'], axis=0)
        for _ in range(4): starts.append(ub + rng0.normal(0, 1.5, n))
        best = None
        for x0 in starts:
            rr = minimize(U, x0, jac=gU, method='L-BFGS-B', options=dict(maxiter=2000))
            if best is None or rr.fun < best[0]: best = (float(rr.fun), rr.x)
        Ustar, thstar = best
        for nm in NAMES:
            if nm == 'acMH': res = acmh(Utau, mode, ini, s, STEP, T, P, L)
            elif nm == 'AWSGLD': res = awsgld(Utau, gUtau, mode, ini, 1000 + s, T, P, L, TAU=TAU, ZETA=ZETA, eps0=EPS0)
            else: res = sgld_family(nm, Utau, gUtau, mode, ini, s, T, P, L, TAU=TAU, base_lr=LR[nm])
            th = res['theta'][BURN:]; pi = inv_logit(th).mean(0); thm = th.mean(0)
            full = res['theta']; Uvals = np.array([U(full[t]) for t in range(0, len(full), 10)])
            bi = int(np.argmin(Uvals)) * 10; bU = float(U(full[bi])); bth = full[bi]
            summ = summarize(res, len(mg['centers']), BURN, im)
            q05, q95 = np.quantile(th, 0.05, 0), np.quantile(th, 0.95, 0)
            rec = dict(basins=summ['visited'], escape=1.0 if summ['escape_iter'] >= 0 else 0.0,
                       spear=spearmanr(pi, ts).correlation, mse=float(np.mean((thm - ts) ** 2)),
                       sig=float(th.std(0).mean()), cov90=float(np.mean((ts >= q05) & (ts <= q95))),
                       auc=auc_roc(pi, truth), ndcg=ndcg(pi, truth),
                       minU=bU, optgap=bU - Ustar, optdist=float(np.linalg.norm(bth - thstar)))
            for g in GAMMAS:
                S = fdr_select(pi, g); tp = np.isin(np.where(S)[0], truth).sum()
                p = tp / max(S.sum(), 1); r = tp / len(truth)
                rec[f'P{g}'] = p; rec[f'R{g}'] = r; rec[f'F{g}'] = 0 if p + r == 0 else 2 * p * r / (p + r)
                rec[f'rFDR{g}'] = 1 - p
            topk_sets[nm].append(set(np.argsort(-pi)[:20]))
            for kk, vv in rec.items(): acc[nm].setdefault(kk, []).append(vv)
        print(f"  seed {s} done", flush=True)
    # Jaccard (시드 간 top-20)
    for nm in NAMES:
        Ts = topk_sets[nm]; js = [len(a & b) / len(a | b) for a, b in itertools.combinations(Ts, 2)]
        acc[nm]['jaccard'] = js if js else [0.0]
    return acc


def main():
    acc = run()
    tag = f"mm_k{KDOC}"
    keys = ['basins', 'escape', 'minU', 'optgap', 'optdist', 'spear', 'auc', 'ndcg', 'mse', 'sig', 'cov90', 'jaccard'] + \
           [f'{p}{g}' for g in GAMMAS for p in ('P', 'R', 'F', 'rFDR')]
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{tag}.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(['metric'] + NAMES + [nm + '_sd' for nm in NAMES])
        for k in keys:
            w.writerow([k] + [f"{np.mean(acc[nm][k]):.4f}" for nm in NAMES] + [f"{np.std(acc[nm][k]):.4f}" for nm in NAMES])
    print(f"\n합성 다봉 K={KDOC} (n={N}, {NSEED}시드)")
    print(f"{'metric':>8} | " + " ".join(f"{nm:>8}" for nm in NAMES))
    for k in ['basins', 'escape', 'minU', 'optgap', 'optdist', 'spear', 'auc', 'cov90']:
        print(f"{k:>8} | " + " ".join(f"{np.mean(acc[nm][k]):>8.3f}" for nm in NAMES))
    print(f"저장: {tag}.csv")


if __name__ == "__main__":
    main()
