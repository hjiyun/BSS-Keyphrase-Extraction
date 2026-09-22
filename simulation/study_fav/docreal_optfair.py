"""실데이터 문서 수(K=1/3/10) 병합 — annealed 최적화 공정비교(SG-MCMC), opt_acmh_fair 스타일.

실제 Hulth 문서 K개를 합집합 어휘로 병합(표준 단봉 에너지, u0=병합 TextRank). Study 3(mixture 다봉)과
달리 단봉·θ*(MAP) 존재. 문서는 고정, 시드는 PU 관측 Y·샘플러 난수만 변화(=100시드로 std).
opt_acmh_fair 의 엔진(make/sg_run/optref/Ubatch) 재사용. SG-MCMC 5종(MH는 O(n²)라 제외).

지표(실데이터라 데이터생성 θ 없음 → Spearman·MSE·cov90 제외):
  최적화: min U / opt.gap(U−U*) / ‖π−π*‖ / π-RMS   (θ*=MAP)
  정확도: Top-k / NDCG / AUC                        (키워드 truth)

사용: python3 docreal_optfair.py <K> [nseed=100]
"""
import os, sys, csv, time, importlib.util
import numpy as np
from numpy.linalg import solve, inv, cholesky
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
import keyphrase_functions_awsgld as kfa
import opt_acmh_fair as of                       # make/sg_run/optref/Ubatch/sigm 재사용
_S2 = "/home/jiyoon/BSS-Keyphrase-Extraction/simulation/study_2"
spec = importlib.util.spec_from_file_location("exp", os.path.join(_S2, "acmh_vs_awsgld_4to10.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)

K     = int(sys.argv[1]) if len(sys.argv) > 1 else 3
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 100
D_DAMP = 0.85
METH = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
DOCSETS = {1: ["1994"], 3: ["1994", "212", "227"],
           10: ["1994", "212", "227", "233", "260", "1982", "2117", "206", "374", "403"]}
_ids = [r[0] for r in csv.reader(open(os.path.join(exp.BASE, "selected_ids.txt")))]
DOCS = DOCSETS.get(K, _ids[:K])                  # 하드코딩 K(1/3/10) 아니면 dense 앞 K개
sigm = of.sigm
G = {dc: exp.build_graph(dc) for dc in DOCS}


def build_merge(docs):
    vocab = sorted(set().union(*[set(G[x]['words']) for x in docs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for x in docs:
        g = G[x]; wm = [idx[w] for w in g['words']]; A[np.ix_(wm, wm)] += g['A']
    np.fill_diagonal(A, 0); keep = A.sum(1) > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    D = np.diag(A.sum(1)); B = np.eye(n) - D_DAMP * solve(D, A).T
    u0 = solve(B, np.ones(n) * (1 - D_DAMP))       # 병합 TextRank = 단일 중심(단봉)
    truth = sorted({idx[G[x]['words'][t]] for x in docs for t in G[x]['truth'] if G[x]['words'][t] in idx})
    return n, B, u0, np.array(truth)


def make_Y(n, truth, u0, seed):
    rng = np.random.RandomState(20260921 + seed); Y = np.zeros(n)
    if len(truth):
        k = max(1, len(truth) // 2)                # PU 관측 = truth 절반
        Y[list(rng.choice(truth, min(k, len(truth)), replace=False))] = 1.0
    a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
    return Y, a0


def acc_metric(bth, bU, Umin, truth, n, pistar):
    gap = bU - Umin; pihat = sigm(bth)
    pidist = float(np.linalg.norm(pihat - pistar)); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2)))
    order = np.argsort(-pihat); nt = len(truth); Tset = set(truth.tolist())
    topk = len(set(order[:nt].tolist()) & Tset) / max(nt, 1)
    rel = np.zeros(n); rel[truth] = 1; disc = 1 / np.log2(np.arange(2, 2 + nt)); ndcg = float((rel[order[:nt]] * disc).sum() / disc.sum())
    pos = nt; neg = n - nt; tp = fp = 0; tpr = [0.]; fpr = [0.]
    for i in order:
        if i in Tset: tp += 1
        else: fp += 1
        tpr.append(tp / pos); fpr.append(fp / max(neg, 1))
    auc = float(np.trapezoid(tpr, fpr))
    return dict(minU=bU, gap=gap, pidist=pidist, pirms=pirms, topk=topk, ndcg=ndcg, auc=auc)


def main():
    docs = DOCS; n, B, u0, truth = build_merge(docs)
    print(f"\n{'#'*66}\n실데이터 문서수 K={K} docs={docs} n={n} truth={len(truth)}  ({NSEED}시드, SG-only)\n{'#'*66}", flush=True)
    csvfn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"docreal_optfair_k{K}_seeds.csv")
    cf = open(csvfn, "w", newline=""); cw = csv.writer(cf)
    cw.writerow(["seed", "method", "minU", "gap", "pidist", "pirms", "topk", "ndcg", "auc", "time"])
    R = {m: [] for m in METH}
    for s in range(NSEED):
        Y, a0 = make_Y(n, truth, u0, s)
        U, gradU, BtB = of.make(n, B, u0, Y, a0)
        P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
        th = u0.copy()
        for _ in range(8000): th = th - 0.05 * gradU(th)
        Umin = U(th); pistar = sigm(th); of.sg_run._umin = Umin
        _, _, reach = of.sg_run("AWSGLD", of.MAXB, n, B, u0, Y, a0, U, gradU, P, Lc, np.random.default_rng(0))
        budget = reach if reach is not None else of.MAXB - 1
        for m in METH:
            t0 = time.time(); bU, bth, _ = of.sg_run(m, budget, n, B, u0, Y, a0, U, gradU, P, Lc, np.random.default_rng(0)); dt = time.time() - t0
            mm = acc_metric(bth, bU, Umin, truth, n, pistar); mm['time'] = dt; R[m].append(mm)
            cw.writerow([s, m, mm['minU'], mm['gap'], mm['pidist'], mm['pirms'], mm['topk'], mm['ndcg'], mm['auc'], dt]); cf.flush()
        if s % 10 == 0: print(f"[seed {s}] budget={budget}", flush=True)

    def ms(m, k, f="{:.3f}"):
        v = [r[k] for r in R[m]]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n실데이터 K={K} 병합 최적화 (SG-MCMC, {NSEED}시드, n={n}, truth={len(truth)})")
    print(f"  {'method':>8} | {'min U':>13} {'opt.gap':>11} {'||π-π*||':>10} {'π-RMS':>11} | {'Top-k':>11} {'NDCG':>11} {'AUC':>11}")
    for m in sorted(METH, key=lambda mm: np.mean([r['gap'] for r in R[mm]])):
        print(f"  {m:>8} | {ms(m,'minU','{:.1f}'):>13} {ms(m,'gap','{:.2f}'):>11} {ms(m,'pidist','{:.2f}'):>10} {ms(m,'pirms','{:.4f}'):>11} | {ms(m,'topk'):>11} {ms(m,'ndcg'):>11} {ms(m,'auc'):>11}")
    print(f"저장: docreal_optfair_k{K}_seeds.csv")


if __name__ == "__main__":
    main()
