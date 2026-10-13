"""실데이터(아마존) K리뷰를 '그냥 한번에 병합' → 단일 그래프·단일 중심 u0(병합 TextRank) annealed 최적화.

원본 amazon_pre.txt 방식(단일 리뷰→단일 그래프)의 K리뷰 병합판. 전 리뷰 텍스트를 union 어휘로 합쳐
하나의 co-occurrence 그래프 A, u0=solve(B,(1-d)1). U=lik+‖B(θ−u0)‖²/(2σ²) = 단봉·중심1개.
더하기(ΣU_k)·혼합(log-sum-exp)과 대조. opt_acmh_fair.make/sg_run 엔진 재사용.

정답 없음 → 지표: min U / opt.gap / π-RMS(θ*=MAP) / time + 방법별 top-12 키워드.
사용: PRODUCT_ID=<pid> python3 amazon_merge_opt.py [K=19] [nseed=100]
"""
import os, sys, csv, time
import numpy as np
from numpy.linalg import solve, inv, cholesky
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC")
import keyphrase_functions_awsgld as kfa
_argv = sys.argv; sys.argv = [sys.argv[0]]
import opt_acmh_fair as of
import amazon_sum_opt as asm                          # load_docs/build 재사용(문서·병합그래프·Y 동일)
sys.argv = _argv

K     = int(sys.argv[1]) if len(sys.argv) > 1 else 19
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 100
D, S2 = 0.85, 1.0
METH = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = of.sigm


def main():
    docs = asm.load_docs(K); n, B, centers, vocab, Y, idx = asm.build(docs)
    u0 = solve(B, np.ones(n) * (1 - D))               # 병합 TextRank = 단일 중심
    a0 = float(kfa.alpha_find(u0, Y, kfa.grid))
    U, gradU, BtB = of.make(n, B, u0, Y, a0)          # 단일중심 단봉 에너지(S2=of.S2)
    th = u0.copy()
    for _ in range(8000): th = th - 0.05 * gradU(th)
    Ustar = U(th); thstar = th.copy(); pistar = sigm(thstar); of.sg_run._umin = Ustar
    print(f"\n{'#'*64}\n아마존 오트밀 '한번에 병합' K={len(docs)}문서  n={n}  Y=1={int(Y.sum())}  α={a0:.3f}"
          f"\n  단일 그래프·단일 u0(병합 TextRank) → 단봉   U*={Ustar:.2f}\n{'#'*64}", flush=True)

    P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
    csvfn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"amazon_merge_k{len(docs)}_seeds.csv")
    cf = open(csvfn, "w", newline=""); cw = csv.writer(cf); cw.writerow(["seed", "method", "minU", "gap", "pirms", "Te", "time"])
    R = {m: [] for m in METH}; best_th = {m: None for m in METH}; best_U = {m: np.inf for m in METH}
    for s in range(NSEED):
        for m in METH:                                   # 예산=고정 MAXB(공정), Tε=EPS 도달 스텝(속도) 별도 기록
            t0 = time.time(); bU, bth, reach = of.sg_run(m, of.MAXB, n, B, u0, Y, a0, U, gradU, P, Lc, np.random.default_rng(s)); dt = time.time() - t0
            pihat = sigm(bth); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2))); Te = reach if reach is not None else of.MAXB
            R[m].append(dict(minU=bU, gap=bU - Ustar, pirms=pirms, Te=Te, time=dt))
            if bU < best_U[m]: best_U[m] = bU; best_th[m] = bth.copy()
            cw.writerow([s, m, bU, bU - Ustar, pirms, Te, dt]); cf.flush()
        if s % 20 == 0: print(f"[seed {s}]", flush=True)

    def ms(m, k, f="{:.3f}"):
        v = [r[k] for r in R[m]]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n아마존 오트밀 '한번에 병합' K={len(docs)}문서 annealed 최적화 (SG-MCMC, {NSEED}시드, n={n}, 단봉, α={a0:.3f}, 예산=고정MAXB={of.MAXB})")
    print(f"  {'method':>8} | {'min U':>14} {'opt.gap':>12} {'π-RMS':>13} {'Tε(속도)':>13} {'time(s)':>12}")
    for m in sorted(METH, key=lambda mm: np.mean([r['gap'] for r in R[mm]])):
        print(f"  {m:>8} | {ms(m,'minU','{:.1f}'):>14} {ms(m,'gap','{:.2f}'):>12} {ms(m,'pirms','{:.4f}'):>13} {ms(m,'Te','{:.0f}'):>13} {ms(m,'time','{:.2f}'):>12}")
    print("\n[top-12 키워드 π̂ 내림차순]  * = 관측 seed")
    order = np.argsort(-pistar); print(f"  {'MAP(θ*)':>8}: {', '.join(vocab[i]+('*' if Y[i] else '') for i in order[:12])}")
    for m in METH:
        o = np.argsort(-sigm(best_th[m])); print(f"  {m:>8}: {', '.join(vocab[i]+('*' if Y[i] else '') for i in o[:12])}")
    print(f"\n저장: {os.path.basename(csvfn)}")


if __name__ == "__main__":
    main()
