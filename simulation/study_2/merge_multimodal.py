"""Hulth 다문서 병합의 자연 다봉성 진단 (합성 트랩 없이).

가설: 문서들을 '어휘를 공유하게' 병합하면 브리지 노드에 상충 인력(frustration)이 생겨
      사후분포가 다봉이 된다. 어휘가 안 겹치면 블록대각 → 분리가능 → 단봉.

구성(원본 모형, 절단 없음):
  병합 A = Σ_doc (문서별 window=2 동시출현 인접행렬을 공유 어휘 인덱스로 매핑해 합산)
  B = I − d·Gᵀ,  u_0 = solve(B,(1−d)1),  truth = 문서별 키워드 합집합
  Y = 문서마다 자기 키워드의 20%를 관측(시드 고정)
에너지(α는 α_find(u_0)로 고정 → U는 θ의 결정함수):
  σ²-explicit  U = −loglik + C/(2σ²),  σ² ∈ {0.5, 1.0}   ← floor 실험과 동일 값
  σ²-적분      U = −loglik + (n/2+ε)·log(C/2+ε)
진단:
  ① 다중 재출발 국소최소화(랜덤 + 문서편향 초기값) → 수렴점 군집 → basin 개수
  ② 서로 다른 basin 쌍 사이 선형보간 U(θ(λ)) → 장벽 높이
조건: HIGH5(고중첩 5편) / HIGH2(최고중첩 2편) / LOW5(저중첩 5편, 대조군)
판정: 고중첩에서 basin>1 & 장벽>0 이고 저중첩에서 basin=1 → "병합이 자연 다봉 생성" 확정.
"""
import os, sys, csv, itertools, importlib.util
import numpy as np
from numpy.linalg import solve
from scipy.optimize import minimize

sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as M
from keyphrase_functions_awsgld import posterior_energy, grad_posterior_energy, alpha_find, inv_logit
from mala_keyphrase import energy_integrated, grad_energy_integrated

spec = importlib.util.spec_from_file_location("exp", "acmh_vs_awsgld_4to10.py")
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
d_damp = 0.85; grid = exp.grid; SEED = 20260721; OBS_RATIO = 0.20
N_RANDOM = 24                      # 랜덤 재출발 수
np.random.seed(SEED)

dense = [r[0] for r in csv.reader(open(os.path.join(exp.BASE, "selected_ids.txt")))]
G_ALL = {}
for dc in dense:
    try: G_ALL[dc] = exp.build_graph(dc)
    except Exception: pass
IDS = list(G_ALL); V = {k: set(G_ALL[k]['words']) for k in IDS}

# ---- 조건별 문서 선정 ----
HIGH5 = ['2010', '2092', '1980', '1994', '1995']
HIGH2 = ['1994', '1995']
# LOW5: 상호 중첩 최소 (그리디)
low = ['1994']
while len(low) < 5:
    cand = min((x for x in IDS if x not in low), key=lambda x: sum(len(V[x] & V[s]) for s in low))
    low.append(cand)
LOW5 = low
CONDS = [('HIGH5', HIGH5), ('HIGH2', HIGH2), ('LOW5', LOW5)]

def merge(docs):
    """문서들을 공유 어휘로 병합. 반환: n, B, BtB, u_0, Y, truth, alpha, bridge수, doc별 truth."""
    vocab = sorted(set().union(*[V[x] for x in docs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for x in docs:                      # 문서 내 동시출현만 합산 (문서 간 인위적 간선 없음)
        g = G_ALL[x]; wmap = [idx[w] for w in g['words']]
        A[np.ix_(wmap, wmap)] += g['A']
    np.fill_diagonal(A, 0)
    deg = A.sum(1)
    keep = deg > 0
    if not keep.all():                  # 고립 노드 제거 (B 특이 방지)
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab); deg = A.sum(1)
    D = np.diag(deg); Bm = np.eye(n) - d_damp * solve(D, A).T
    u_0 = solve(Bm, np.ones(n) * (1 - d_damp))
    truth_doc = {}
    for x in docs:
        g = G_ALL[x]
        truth_doc[x] = sorted({idx[g['words'][t]] for t in g['truth'] if g['words'][t] in idx})
    truth = sorted(set().union(*truth_doc.values()))
    rng = np.random.RandomState(SEED)
    Y = np.zeros(n)
    for x in docs:                      # 문서마다 자기 키워드의 20% 관측
        td = truth_doc[x]
        if td:
            k = max(1, round(OBS_RATIO * len(td)))
            Y[list(rng.choice(td, min(k, len(td)), replace=False))] = 1
    cnt = {}
    for x in docs:
        for w in V[x]: cnt[w] = cnt.get(w, 0) + 1
    bridge = sum(1 for w in vocab if cnt.get(w, 0) >= 2)
    alpha = alpha_find(u_0, Y, grid)
    return dict(n=n, B=Bm, BtB=Bm.T @ Bm, u_0=u_0, Y=Y, truth=truth,
                alpha=alpha, bridge=bridge, truth_doc=truth_doc, vocab=vocab)

def make_UF(mg, mode):
    """U(θ), ∇U(θ) 반환. mode='s0.5'|'s1.0'|'marg' (α 고정 → 결정함수)."""
    Y, u0, Bm, BtB, a = mg['Y'], mg['u_0'], mg['B'], mg['BtB'], mg['alpha']
    if mode == 'marg':
        return (lambda th: float(energy_integrated(Y, a, th, u0, Bm)),
                lambda th: np.asarray(grad_energy_integrated(Y, a, th, u0, Bm, BtB=BtB), float))
    s2 = float(mode[1:])
    return (lambda th: float(posterior_energy(Y, a, th, u0, Bm, s2)),
            lambda th: np.asarray(grad_posterior_energy(Y, a, th, u0, Bm, s2, BtB=BtB), float))

def local_min(U, gU, x0):
    r = minimize(U, x0, jac=gU, method='L-BFGS-B',
                 options=dict(maxiter=4000, ftol=1e-12, gtol=1e-9))
    return r.x, float(r.fun)

def basins(mg, mode):
    """다중 재출발 → 수렴점 군집 → (basin 대표들, 에너지들)."""
    U, gU = make_UF(mg, mode); n = mg['n']; u0 = mg['u_0']
    starts = []
    rng = np.random.RandomState(SEED + 1)
    for s in (0.5, 1.0, 2.0):                       # 랜덤 재출발
        for _ in range(N_RANDOM // 3):
            starts.append(u0 + rng.normal(0, s, n))
    for x, td in mg['truth_doc'].items():           # 문서편향 초기값 ('그 문서 관점의 해')
        z = u0.copy(); z[td] += 3.0; starts.append(z)
    mins = []
    for x0 in starts:
        xm, fm = local_min(U, gU, x0)
        mins.append((fm, xm))
    # 군집: 거리 (정규화) < 0.05 이면 동일 basin
    reps = []
    for fm, xm in sorted(mins, key=lambda t: t[0]):
        if all(np.linalg.norm(xm - r[1]) / np.sqrt(n) >= 0.05 for r in reps):
            reps.append((fm, xm))
    return reps

def barrier(mg, mode, xa, xb, npt=41):
    U, _ = make_UF(mg, mode)
    lams = np.linspace(0, 1, npt)
    vals = np.array([U((1 - l) * xa + l * xb) for l in lams])
    return float(vals.max() - max(vals[0], vals[-1])), vals

print(f"Hulth 병합 다봉성 진단 | seed={SEED}, 관측비율={OBS_RATIO}, 재출발={N_RANDOM}+문서편향\n")
print(f"{'조건':>7} {'문서수':>5} {'병합n':>6} {'브리지':>6} {'비율':>6} | "
      f"{'모드':>6} {'basin수':>7} {'최저E':>11} {'2nd-E차':>9} {'최대장벽':>9}")
print("-" * 96)
OUT = []
for cname, docs in CONDS:
    mg = merge(docs)
    for mode in ('s0.5', 's1.0', 'marg'):
        reps = basins(mg, mode)
        nb = len(reps)
        e0 = reps[0][0]; de = (reps[1][0] - e0) if nb > 1 else float('nan')
        bmax = 0.0
        for i, j in itertools.combinations(range(min(nb, 4)), 2):
            b, _ = barrier(mg, mode, reps[i][1], reps[j][1])
            bmax = max(bmax, b)
        print(f"{cname:>7} {len(docs):>5} {mg['n']:>6} {mg['bridge']:>6} "
              f"{mg['bridge']/mg['n']*100:>5.1f}% | {mode:>6} {nb:>7} {e0:>11.2f} "
              f"{de:>9.2f} {bmax:>9.3f}")
        OUT.append([cname, len(docs), mg['n'], mg['bridge'], mode, nb, e0, de, bmax])
    print("-" * 96)

with open("merge_multimodal.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["cond", "ndocs", "n", "bridge", "mode", "basins", "E_min", "dE_2nd", "max_barrier"])
    w.writerows([[a, b, c, d, e, g, round(h, 3), (round(i, 3) if i == i else ""), round(j, 4)]
                 for a, b, c, d, e, g, h, i, j in OUT])
print("\n판정 기준: 고중첩(HIGH5/HIGH2)에서 basin>1 & 장벽>0, 저중첩(LOW5)에서 basin=1 → 병합이 자연 다봉 생성")
print("저장: merge_multimodal.csv")
