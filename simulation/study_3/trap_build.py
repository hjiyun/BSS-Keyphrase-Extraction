"""실데이터 기반 로컬 트랩 구성 + 검증 (중심을 지어내지 않음).

구성:
  Hulth 문서 K개 → 합집합 어휘로 통합(공유 stem = 같은 노드)
  각 문서 k 의 TextRank 해 u^(k) = solve(B_k,(1-d)1) 를 합집합 어휘에 배치
    (문서 k 에 없는 단어는 baseline) → 문서 k 봉우리의 중심. 전부 실데이터 산출값.
  공유 그래프행렬 B = 병합 그래프에서 계산
  U_k(θ) = -loglik(θ) + ||B(θ-u^(k))||^2/(2σ²)
  U_mix(θ) = -logsumexp(-U_k(θ))            ← 다봉 (CLAUDE.md 규칙)

검증: 다중 재출발 국소최소화 → basin 개수 / 봉우리 간 장벽 높이 / 중심 간 거리
문서 조합은 '중첩 낮음(중심이 멀다=트랩 깊다)' 우선으로 탐색.
"""
import os, sys, csv, itertools, importlib.util
import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize
from scipy.special import logsumexp

sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
from keyphrase_functions_awsgld import alpha_find, inv_logit

_S2 = "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_2"
sys.path.insert(0, _S2)
spec = importlib.util.spec_from_file_location("exp", os.path.join(_S2, "acmh_vs_awsgld_4to10.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
d_damp = 0.85; grid = exp.grid; SEED = 20260721
BASELINE = -0.5          # 문서 k 에 없는 단어의 중심값
OBS_RATIO = 0.20
np.random.seed(SEED)

dense = [r[0] for r in csv.reader(open(os.path.join(exp.BASE, "selected_ids.txt")))]
G = {}
for dc in dense:
    try: G[dc] = exp.build_graph(dc)
    except Exception: pass
IDS = list(G); V = {k: set(G[k]['words']) for k in IDS}

def textrank_center(doc):
    """문서 자체 그래프의 TextRank 해 (실데이터 산출)."""
    g = G[doc]; nk = g['n']
    Bk = np.eye(nk) - d_damp * solve(g['D'], g['A']).T
    return solve(Bk, np.ones(nk) * (1 - d_damp))

def build(docs, sigma2=2.0):
    vocab = sorted(set().union(*[V[x] for x in docs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for x in docs:
        g = G[x]; wm = [idx[w] for w in g['words']]
        A[np.ix_(wm, wm)] += g['A']
    np.fill_diagonal(A, 0)
    keep = A.sum(1) > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    D = np.diag(A.sum(1)); B = np.eye(n) - d_damp * solve(D, A).T
    # 문서별 중심 (실 TextRank 를 합집합 어휘에 배치)
    centers = []
    for x in docs:
        g = G[x]; tr = textrank_center(x)
        u = np.full(n, BASELINE, float)
        for j, w in enumerate(g['words']):
            if w in idx: u[idx[w]] = tr[j]
        centers.append(u)
    truth_doc = {x: sorted({idx[G[x]['words'][t]] for t in G[x]['truth'] if G[x]['words'][t] in idx}) for x in docs}
    truth = sorted(set().union(*truth_doc.values()))
    rng = np.random.RandomState(SEED); Y = np.zeros(n)
    for x in docs:
        td = truth_doc[x]
        if td:
            k = max(1, round(OBS_RATIO * len(td)))
            Y[list(rng.choice(td, min(k, len(td)), replace=False))] = 1
    u_bar = np.mean(centers, axis=0)
    alpha = alpha_find(u_bar, Y, grid)
    return dict(n=n, B=B, BtB=B.T @ B, centers=centers, Y=Y, truth=truth,
                truth_doc=truth_doc, alpha=alpha, sigma2=sigma2, vocab=vocab, docs=docs)

def energies(mg):
    Y, B, BtB, a, s2 = mg['Y'], mg['B'], mg['BtB'], mg['alpha'], mg['sigma2']
    cs = mg['centers']
    def lik(th):
        t = np.clip((1 - a) * inv_logit(th), 1e-10, 1 - 1e-10)
        return -float(np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t)))
    def glik(th):
        p = np.clip(inv_logit(th), 1e-10, 1 - 1e-10); dp = p * (1 - p)
        t = np.clip((1 - a) * p, 1e-10, 1 - 1e-10)
        return -(Y * (1 - a) * dp / t - (1 - Y) * (1 - a) * dp / (1 - t))
    def Uk(th, u):
        r = B @ (th - u); return lik(th) + float(r @ r) / (2 * s2)
    def Umix(th): return -float(logsumexp([-Uk(th, u) for u in cs]))
    def gUmix(th):
        Us = np.array([-Uk(th, u) for u in cs]); w = np.exp(Us - logsumexp(Us))
        g = np.zeros_like(th)
        for wi, u in zip(w, cs):
            g += wi * (glik(th) + BtB @ (th - u) / s2)
        return g
    def mode(th): return int(np.argmin([Uk(th, u) for u in cs]))
    return Umix, gUmix, mode

def diagnose(mg, nrand=12):
    U, gU, mode = energies(mg); n = mg['n']
    starts = [c.copy() for c in mg['centers']]            # 각 문서 중심에서 출발
    rng = np.random.RandomState(SEED + 3)
    ub = np.mean(mg['centers'], axis=0)
    for s in (0.5, 1.5):
        for _ in range(nrand // 2): starts.append(ub + rng.normal(0, s, n))
    mins = []
    for x0 in starts:
        r = minimize(U, x0, jac=gU, method='L-BFGS-B', options=dict(maxiter=3000, ftol=1e-12, gtol=1e-9))
        mins.append((float(r.fun), r.x))
    reps = []
    for f, x in sorted(mins, key=lambda t: t[0]):
        if all(np.linalg.norm(x - r[1]) / np.sqrt(n) >= 0.05 for r in reps): reps.append((f, x))
    bmax = 0.0; bmin = None
    for i, j in itertools.combinations(range(min(len(reps), 5)), 2):
        vals = np.array([U((1 - l) * reps[i][1] + l * reps[j][1]) for l in np.linspace(0, 1, 41)])
        b = float(vals.max() - max(vals[0], vals[-1])); bmax = max(bmax, b)
        bmin = b if bmin is None else min(bmin, b)
    cd = np.mean([np.linalg.norm(a - b) / np.sqrt(mg['n'])
                  for a, b in itertools.combinations(mg['centers'], 2)])
    return len(reps), bmax, (bmin if bmin is not None else 0.0), cd, reps

# ---- 문서 조합 후보: 중첩 낮은(=중심이 먼) 조합 우선 ----
def pick_low(K, seed_doc='1994'):
    sel = [seed_doc]
    while len(sel) < K:
        cand = min((x for x in IDS if x not in sel), key=lambda x: sum(len(V[x] & V[s]) for s in sel))
        sel.append(cand)
    return sel

def _scan():
    """파라미터 스캔 (직접 실행할 때만; 임포트 시에는 실행 안 함)."""
    print(f"실데이터 로컬 트랩 구성/검증 | 중심=문서별 TextRank(실산출), baseline={BASELINE}\n")
    print(f"{'K':>2} {'σ²':>5} {'n':>5} {'중심간거리':>9} {'basin수':>7} {'최대장벽':>9} {'최소장벽':>9} {'문서':<32}")
    print("-" * 96)
    ROWS = []
    for K in (3, 5):
        docs = pick_low(K)
        for s2 in (1.0, 2.0, 4.0):
            mg = build(docs, sigma2=s2)
            nb, bmax, bmin, cd, reps = diagnose(mg)
            print(f"{K:>2} {s2:>5.1f} {mg['n']:>5} {cd:>9.3f} {nb:>7} {bmax:>9.3f} "
                  f"{bmin:>9.3f} {','.join(docs):<32}")
            ROWS.append([K, s2, mg['n'], round(cd, 4), nb, round(bmax, 4), round(bmin, 4), " ".join(docs)])
    print("-" * 96)
    print("판정: basin수 = 문서수 이고 최소장벽 > 0 이면 '실데이터 로컬 트랩 구성 성공'")
    with open("real_trap_build.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["K", "sigma2", "n", "center_dist", "basins", "barrier_max", "barrier_min", "docs"])
        w.writerows(ROWS)
    print("저장: real_trap_build.csv")


if __name__ == "__main__":
    _scan()
