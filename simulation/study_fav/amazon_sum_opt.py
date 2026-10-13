"""실데이터(아마존 오트밀 리뷰) K개 문서를 '그냥 더하기' ΣU_k 로 결합해 annealed 최적화.

docsum_optfair(합성/Hulth) 의 실데이터판. 정답(gold) 없음 → 정확도(Top-k/AUC) 제외.
문서별 에너지 U_k = K·lik + ‖B(θ−u^k)‖²/(2σ²) 를 더함 → 단봉(볼록) 예상, basin 진단으로 확인.
관측 seed Y=1 = 각 리뷰 Summary 전처리 토큰(병합 어휘에 존재하는 것)의 합집합.

지표: min U / opt.gap(U−U*) / π-RMS(‖π−π*‖/√n, θ*=MAP) / time  + basin 수 + 방법별 top-12 키워드.

사용: python3 amazon_sum_opt.py [K=30] [nseed=30]
"""
import os, sys, csv, time
import numpy as np
from numpy.linalg import solve, inv, cholesky
from scipy.optimize import minimize
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC")
import keyphrase_functions_awsgld as kfa
from preprocess_amazon import preprocess
_argv = sys.argv; sys.argv = [sys.argv[0]]
import opt_acmh_fair as of
sys.argv = _argv

K     = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 30
D, BASELINE, S2, MINTOK = 0.85, -2.0, 0.5, 30
REVIEWS = os.environ.get("REVIEWS", "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC/amazon/Reviews.csv")
PRODUCT_ID = os.environ.get("PRODUCT_ID", "")       # 지정 시 그 상품의 전 리뷰만(K 무시), 미지정 시 oatmeal 카테고리
METH = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = of.sigm


def load_docs(k):
    """PRODUCT_ID 지정 시 그 상품의 전 리뷰(Id순); 아니면 oatmeal Summary·토큰수>=MINTOK 앞 k개."""
    csv.field_size_limit(10 ** 7); rows = []
    with open(REVIEWS, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if PRODUCT_ID:
                if row["ProductId"] == PRODUCT_ID:
                    pp = preprocess(row["Text"])
                    if pp.strip():
                        rows.append((int(row["Id"]), pp, preprocess(row["Summary"])))
            elif "oatmeal" in row["Summary"].lower():
                pp = preprocess(row["Text"])
                if len(pp.split()) >= MINTOK:
                    rows.append((int(row["Id"]), pp, preprocess(row["Summary"])))
    rows.sort(key=lambda r: r[0])
    return rows if PRODUCT_ID else rows[:k]


def doc_graph(text):
    A0, words, _ = kfa.create_fcm_words(text, window=2)
    A0 = np.array(A0, float); np.fill_diagonal(A0, 0)
    deg = A0.sum(1); keep = deg > 0
    if not keep.all():
        A0 = A0[np.ix_(keep, keep)]; words = [w for w, k in zip(words, keep) if k]
    return A0, words


def build(docs):
    graphs = [doc_graph(t) for _id, t, _s in docs]
    vocab = sorted(set().union(*[set(w) for _, w in graphs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for Ak, wk in graphs:
        m = [idx[w] for w in wk]; A[np.ix_(m, m)] += Ak
    np.fill_diagonal(A, 0); keep = A.sum(1) > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    Dm = np.diag(A.sum(1)); B = np.eye(n) - D * solve(Dm, A).T
    centers = []
    for Ak, wk in graphs:
        dk = Ak.sum(1); Bk = np.eye(len(wk)) - D * solve(np.diag(dk), Ak).T
        trk = solve(Bk, np.ones(len(wk)) * (1 - D)); u = np.full(n, BASELINE, float)
        for j, w in enumerate(wk):
            if w in idx: u[idx[w]] = trk[j]
        centers.append(u)
    Y = np.zeros(n)
    import collections
    cnt = collections.Counter()
    for _id, _t, summ in docs:
        for w in set(summ.split()):
            if w in idx: cnt[idx[w]] += 1          # 토큰이 등장한 리뷰 제목 수
    _frac = os.environ.get("OBSFRAC")
    if _frac:                                       # 관측을 n의 일정 비율로 제한(빈도 상위)
        cap = max(1, round(float(_frac) * n))
        for i, _ in cnt.most_common(cap): Y[i] = 1.0
    else:
        for i in cnt: Y[i] = 1.0
    return n, B, centers, np.array(vocab), Y, idx


def energies(n, B, centers, Y, alpha):
    BtB = B.T @ B; Kc = len(centers); Su = np.sum(centers, 0)
    def lik(th):
        t = np.clip((1 - alpha) * sigm(th), 1e-10, 1 - 1e-10); return -float(np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t)))
    def glik(th):
        p = np.clip(sigm(th), 1e-10, 1 - 1e-10); dp = p * (1 - p); t = np.clip((1 - alpha) * p, 1e-10, 1 - 1e-10)
        return -((1 - alpha) * dp) * (Y / t - (1 - Y) / (1 - t))
    def Usum(th):
        pr = sum(float((B @ (th - u)) @ (B @ (th - u))) for u in centers) / (2 * S2)
        return Kc * lik(th) + pr
    def gUsum(th):
        return Kc * glik(th) + (BtB @ (Kc * th - Su)) / S2
    return Usum, gUsum, BtB


def gmin(Usum, gUsum, n, centers):
    starts = [c.copy() for c in centers] + [np.mean(centers, 0)]; rng = np.random.RandomState(7)
    for _ in range(8): starts.append(np.mean(centers, 0) + rng.normal(0, 1.5, n))
    res = []
    for x0 in starts:
        r = minimize(Usum, x0, jac=gUsum, method="L-BFGS-B", options=dict(maxiter=3000, ftol=1e-12)); res.append((float(r.fun), r.x))
    reps = []
    for f, x in sorted(res, key=lambda t: t[0]):
        if all(np.linalg.norm(x - r[1]) / np.sqrt(n) >= 0.05 for r in reps): reps.append((f, x))
    return reps[0][0], reps[0][1], len(reps)


def main():
    docs = load_docs(K); n, B, centers, vocab, Y, idx = build(docs)
    a0 = float(kfa.alpha_find(np.mean(centers, 0), Y, kfa.grid))
    Us, gUs, BtB = energies(n, B, centers, Y, a0)
    Ustar, thstar, nb = gmin(Us, gUs, n, centers); of.sg_run._umin = Ustar
    pistar = sigm(thstar)
    print(f"\n{'#'*64}\n아마존 오트밀 ΣU_k(더하기) K={len(docs)}문서  n={n}  관측seed(Y=1)={int(Y.sum())}  α={a0:.3f}"
          f"\n  basin={nb}개 → {'단봉' if nb==1 else f'다봉{nb}'}   U*={Ustar:.2f}\n{'#'*64}", flush=True)

    P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
    csvfn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"amazon_sum_k{len(docs)}_seeds.csv")
    cf = open(csvfn, "w", newline=""); cw = csv.writer(cf); cw.writerow(["seed", "method", "minU", "gap", "pirms", "Te", "time"])
    R = {m: [] for m in METH}; best_th = {m: None for m in METH}; best_U = {m: np.inf for m in METH}
    for s in range(NSEED):
        for m in METH:                                   # 예산=고정 MAXB(공정), Tε=EPS 도달 스텝(속도) 별도 기록
            t0 = time.time(); bU, bth, reach = of.sg_run(m, of.MAXB, n, B, centers[0], Y, a0, Us, gUs, P, Lc, np.random.default_rng(s)); dt = time.time() - t0
            pihat = sigm(bth); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2))); Te = reach if reach is not None else of.MAXB
            R[m].append(dict(minU=bU, gap=bU - Ustar, pirms=pirms, Te=Te, time=dt))
            if bU < best_U[m]: best_U[m] = bU; best_th[m] = bth.copy()
            cw.writerow([s, m, bU, bU - Ustar, pirms, Te, dt]); cf.flush()
        if s % 10 == 0: print(f"[seed {s}]", flush=True)

    def ms(m, k, f="{:.3f}"):
        v = [r[k] for r in R[m]]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n아마존 오트밀 ΣU_k(더하기) K={len(docs)}문서 annealed 최적화 (SG-MCMC, {NSEED}시드, n={n}, basin={nb})")
    print(f"  {'method':>8} | {'min U':>14} {'opt.gap':>12} {'π-RMS':>13} {'Tε(속도)':>13} {'time(s)':>12}")
    for m in sorted(METH, key=lambda mm: np.mean([r['gap'] for r in R[mm]])):
        print(f"  {m:>8} | {ms(m,'minU','{:.1f}'):>14} {ms(m,'gap','{:.2f}'):>12} {ms(m,'pirms','{:.4f}'):>13} {ms(m,'Te','{:.0f}'):>13} {ms(m,'time','{:.2f}'):>12}")

    seed_words = sorted({w for w in vocab[Y.astype(bool)]})
    print(f"\n관측 seed 단어({len(seed_words)}개): {', '.join(seed_words[:25])}{' ...' if len(seed_words)>25 else ''}")
    print("\n[top-12 키워드 π̂=sigmoid(θ̂) 내림차순]  * = 관측 seed")
    order = np.argsort(-pistar); toks = [vocab[i] + ("*" if Y[i] else "") for i in order[:12]]
    print(f"  {'MAP(θ*)':>8}: {', '.join(toks)}")
    for m in METH:
        o = np.argsort(-sigm(best_th[m])); tk = [vocab[i] + ("*" if Y[i] else "") for i in o[:12]]
        print(f"  {m:>8}: {', '.join(tk)}")
    print(f"\n저장: {os.path.basename(csvfn)}")


if __name__ == "__main__":
    main()
