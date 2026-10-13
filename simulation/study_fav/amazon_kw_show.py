"""아마존 단일상품 키프레이즈(구) 추출 표시 — 단어-노드 BSS + 인접 고-π 단어 구 조립.
env: REVIEWS, PRODUCT_ID, LABEL. 각 방법 top-15 다단어 키프레이즈(점수순)."""
import os, sys
import numpy as np
from numpy.linalg import solve, inv, cholesky
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_fav")
import keyphrase_functions_awsgld as kfa
_a = sys.argv; sys.argv = [sys.argv[0]]
import opt_acmh_fair as of
import amazon_sum_opt as asm
sys.argv = _a
of.LR_SGLD = 0.6; of.LR_QSGLD = 0.6; of.LR_CYC = 0.05; of.LR_SGHMC = 0.05; of.LR_AWSGLD = 0.6
of.MAXB = 3000; of.TAU = 400.0
D = 0.85; sigm = of.sigm; SELFRAC = 1.0/3.0; LABEL = os.environ.get("LABEL", "")
METH = ["AWSGLD", "qSGLD", "cycSGLD", "SGHMC", "SGLD"]

docs = asm.load_docs(999)
seqs = [t.split() for (_id, t, _s) in docs]
n, B, centers, vocab, Y, idx = asm.build(docs)
u0 = solve(B, np.ones(n) * (1 - D)); a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
U, gradU, BtB = of.make(n, B, u0, Y, a0)
th = u0.copy()
for _ in range(8000): th = th - 0.05 * gradU(th)
of.sg_run._umin = U(th)
P = inv(BtB + 1e-6*np.trace(BtB)/n*np.eye(n)); P = 0.5*(P+P.T); Lc = cholesky(P + 1e-10*np.eye(n))
T = max(1, round(SELFRAC * n))


def multiword(pihat, selset):
    out = {}   # key=frozenset(words) → (phrase, score)  순서무관 중복제거
    for s in seqs:
        L = len(s)
        for i in range(L):
            if s[i] not in idx or idx[s[i]] not in selset: continue
            words = []
            for j in range(i, min(i + 4, L)):
                if s[j] not in idx or idx[s[j]] not in selset: break
                if s[j] in words: break
                words.append(s[j])
                if len(words) >= 2:
                    key = frozenset(words); sc = float(np.mean([pihat[idx[w]] for w in words]))
                    if key not in out or sc > out[key][1]: out[key] = (" ".join(words), sc)
    return sorted(out.values(), key=lambda x: -x[1])


print(f"\n########## {LABEL}  n(단어)={n}  리뷰 {len(docs)}건 ##########")
top = {}
for m in METH:
    bU, bth, _ = of.sg_run(m, of.MAXB, n, B, u0, Y, a0, U, gradU, P, Lc, np.random.default_rng(0))
    pihat = sigm(bth); selset = set(np.argsort(-pihat)[:T].tolist())
    cands = multiword(pihat, selset)
    top[m] = cands[:15]
    print(f"\n[{m}] opt.gap={bU-U(th):.3f}  상위 15 다단어 키프레이즈:")
    for p, sc in cands[:15]:
        print(f"    {sc:.4f}  {p}")
# 세 preconditioned 동일성 확인
def phr(lst): return [p for p, _ in lst]
aw, q, c = phr(top["AWSGLD"]), phr(top["qSGLD"]), phr(top["cycSGLD"])
print(f"\n[동일성] AWSGLD vs qSGLD: 순서까지 동일={aw==q}, 집합 동일={set(aw)==set(q)}")
print(f"         AWSGLD vs cycSGLD: 순서까지 동일={aw==c}, 집합 동일={set(aw)==set(c)}")
