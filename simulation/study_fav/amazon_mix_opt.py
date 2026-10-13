"""실데이터(아마존 오트밀) K문서 log-sum-exp '혼합(다봉)' → annealed 최적화. amazon_sum_opt 의 mixture 대조군.

U_mix(θ)=−logΣ_k exp(−U_k),  U_k=lik(θ)+‖B(θ−u^k)‖²/(2σ²).  문서별 TextRank u^(k)=봉우리 → 다봉(굴곡).
더하기(ΣU_k, 단봉)와 달리 basin 여러 개 → annealed 최적화가 가장 깊은 basin 찾는지 + AWSGLD 트랩탈출 비교.
정답 없음 → 지표: min U / opt.gap(전역 U* 대비) / π-RMS / 방문 basin / time + 방법별 top-12 키워드.

사용: python3 amazon_mix_opt.py [K=30] [nseed=10]
"""
import os, sys, csv, time
import numpy as np
from numpy.linalg import inv, cholesky
from scipy.special import logsumexp
from scipy.optimize import minimize
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/code_JOC")
sys.path.insert(0, "/home/jiyoon/BSS-Keyphrase-Extraction/data_JOC")
import keyphrase_functions_awsgld as kfa
_argv = sys.argv; sys.argv = [sys.argv[0]]
import opt_acmh_fair as of
import amazon_sum_opt as asm                        # load_docs/doc_graph/build 재사용
sys.argv = _argv

K     = int(sys.argv[1]) if len(sys.argv) > 1 else 30
NSEED = int(sys.argv[2]) if len(sys.argv) > 2 else 10
S2 = 0.5
METH = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
sigm = of.sigm


def energies(n, B, centers, Y, alpha):
    BtB = B.T @ B
    def lik(th):
        t = np.clip((1 - alpha) * sigm(th), 1e-10, 1 - 1e-10); return -float(np.sum(Y * np.log(t) + (1 - Y) * np.log(1 - t)))
    def glik(th):
        p = np.clip(sigm(th), 1e-10, 1 - 1e-10); dp = p * (1 - p); t = np.clip((1 - alpha) * p, 1e-10, 1 - 1e-10)
        return -((1 - alpha) * dp) * (Y / t - (1 - Y) / (1 - t))
    def Uk(th, u): r = B @ (th - u); return lik(th) + float(r @ r) / (2 * S2)
    def Umix(th): return -float(logsumexp([-Uk(th, u) for u in centers]))
    def gUmix(th):
        Us = np.array([-Uk(th, u) for u in centers]); w = np.exp(Us - logsumexp(Us)); g = np.zeros(n)
        for wi, u in zip(w, centers): g += wi * (glik(th) + BtB @ (th - u) / S2)
        return g
    def mode(th): return int(np.argmin([Uk(th, u) for u in centers]))
    return Umix, gUmix, mode, BtB


def global_min(Umix, gUmix, n, centers):
    starts = [c.copy() for c in centers]; rng = np.random.RandomState(7); ub = np.mean(centers, 0)
    for _ in range(8): starts.append(ub + rng.normal(0, 1.5, n))
    res = []
    for x0 in starts:
        r = minimize(Umix, x0, jac=gUmix, method="L-BFGS-B", options=dict(maxiter=2000, ftol=1e-11)); res.append((float(r.fun), r.x))
    reps = []
    for f, x in sorted(res, key=lambda t: t[0]):
        if all(np.linalg.norm(x - r[1]) / np.sqrt(n) >= 0.05 for r in reps): reps.append((f, x))
    return reps[0][0], reps[0][1], len(reps)


def main():
    docs = asm.load_docs(K); n, B, centers, vocab, Y, idx = asm.build(docs)
    a0 = float(kfa.alpha_find(np.mean(centers, 0), Y, kfa.grid))
    Umix, gUmix, mode, BtB = energies(n, B, centers, Y, a0)
    Ustar, thstar, nb = global_min(Umix, gUmix, n, centers); of.sg_run._umin = Ustar
    pistar = sigm(thstar)
    print(f"\n{'#'*64}\n아마존 오트밀 log-sum-exp 혼합 K={len(docs)}문서  n={n}  Y=1={int(Y.sum())}  α={a0:.3f}"
          f"\n  basin={nb}개 → {'단봉' if nb==1 else f'다봉{nb}'}   U*={Ustar:.2f}  (전역최소 basin={mode(thstar)})\n{'#'*64}", flush=True)

    P = inv(BtB + 1e-6 * np.trace(BtB) / n * np.eye(n)); P = 0.5 * (P + P.T); Lc = cholesky(P + 1e-10 * np.eye(n))
    csvfn = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"amazon_mix_k{len(docs)}_seeds.csv")
    cf = open(csvfn, "w", newline=""); cw = csv.writer(cf); cw.writerow(["seed", "method", "minU", "gap", "pirms", "basin", "Te", "time"])
    R = {m: [] for m in METH}; best_th = {m: None for m in METH}; best_U = {m: np.inf for m in METH}
    for s in range(NSEED):
        for m in METH:                                   # 예산=고정 MAXB(공정), Tε=EPS 도달 스텝(속도) 별도 기록
            t0 = time.time(); bU, bth, reach = of.sg_run(m, of.MAXB, n, B, centers[0], Y, a0, Umix, gUmix, P, Lc, np.random.default_rng(s)); dt = time.time() - t0
            pihat = sigm(bth); pirms = float(np.sqrt(np.mean((pihat - pistar) ** 2))); bs = mode(bth); Te = reach if reach is not None else of.MAXB
            R[m].append(dict(minU=bU, gap=bU - Ustar, pirms=pirms, basin=bs, Te=Te, time=dt))
            if bU < best_U[m]: best_U[m] = bU; best_th[m] = bth.copy()
            cw.writerow([s, m, bU, bU - Ustar, pirms, bs, Te, dt]); cf.flush()
        if s % 5 == 0: print(f"[seed {s}]", flush=True)

    def ms(m, k, f="{:.3f}"):
        v = [r[k] for r in R[m]]; return f.format(np.mean(v)) + "±" + f.format(np.std(v))
    print(f"\n아마존 오트밀 log-sum-exp 혼합 K={len(docs)}문서 annealed 최적화 (SG-MCMC, {NSEED}시드, n={n}, basin={nb})")
    print(f"  {'method':>8} | {'min U':>14} {'opt.gap':>12} {'π-RMS':>13} {'전역basin도달':>14} {'Tε(속도)':>13} {'time(s)':>12}")
    for m in sorted(METH, key=lambda mm: np.mean([r['gap'] for r in R[mm]])):
        gstar = mode(thstar); hit = np.mean([r['basin'] == gstar for r in R[m]])
        print(f"  {m:>8} | {ms(m,'minU','{:.1f}'):>14} {ms(m,'gap','{:.2f}'):>12} {ms(m,'pirms','{:.4f}'):>13} {hit*100:>12.0f}% {ms(m,'Te','{:.0f}'):>13} {ms(m,'time','{:.2f}'):>12}")
    print("\n[방법별 방문 basin 분포 (basin_id: 횟수)]")
    for m in METH:
        from collections import Counter; c = Counter(r['basin'] for r in R[m])
        print(f"  {m:>8}: {dict(sorted(c.items()))}")
    print("\n[top-12 키워드 π̂ 내림차순]  * = 관측 seed")
    order = np.argsort(-pistar); print(f"  {'MAP(θ*)':>8}: {', '.join(vocab[i]+('*' if Y[i] else '') for i in order[:12])}")
    for m in METH:
        o = np.argsort(-sigm(best_th[m])); print(f"  {m:>8}: {', '.join(vocab[i]+('*' if Y[i] else '') for i in o[:12])}")
    print(f"\n저장: {os.path.basename(csvfn)}")


if __name__ == "__main__":
    main()
