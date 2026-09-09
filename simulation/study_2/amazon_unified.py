"""정답 없는 실데이터(아마존 리뷰 1편) — 6샘플러 비교. 논문 §1.6.2 세팅.
관측 키프레이즈 = 요약("oatmeal for oatmeal lovers") → seed 'oatmeal','love'.
gold 없음 → 수렴·효율(R̂·ESS·LowU) + 샘플러별 top-k 키워드(정성). 논문 Table 1.8 대응.
"""
import os, sys
import numpy as np
from numpy.linalg import solve

_HERE = os.path.dirname(os.path.abspath(__file__))
_1C = os.path.join(os.path.dirname(_HERE), "study_1c")
_A0 = os.path.join(os.path.dirname(_HERE), "study_a0")
ROOT = os.path.dirname(os.path.dirname(_HERE)); CODE = os.path.join(ROOT, "code_JOC")
for p in (_1C, _A0, os.path.join(_A0, "_archive"), CODE):
    sys.path.insert(0, p)
import keyphrase_functions_awsgld as M          # noqa: E402
import energy_diagnostics as E                  # acMH·alpha_find·energy_trace  # noqa: E402
import unified_scaling as US                    # 연속 샘플러  # noqa: E402

d_damp = 0.85
SEED_WORDS = ["oatmeal", "love"]                # 요약에서 온 관측 키프레이즈


def build():
    text = open(os.path.join(ROOT, "data_JOC", "amazon_pre.txt")).read()
    fcm, words, w2i = M.create_fcm_words(text, window=2)
    A = np.array(fcm, float); np.fill_diagonal(A, 0)
    deg = A.sum(1); keep = deg > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; words = [w for w, k in zip(words, keep) if k]
        w2i = {w: i for i, w in enumerate(words)}; deg = A.sum(1)
    n = len(words); D = np.diag(deg)
    B = np.eye(n) - d_damp * solve(D, A).T
    u_0 = solve(B, np.ones(n) * (1 - d_damp))
    Y = np.zeros(n)
    for sw in SEED_WORDS:
        if sw in w2i:
            Y[w2i[sw]] = 1
    a0 = M.alpha_find(u_0, Y, M.grid)
    return dict(n=n, A=A, D=D, B=B, u_0=u_0, Y=Y, words=np.array(words), a0=a0)


def main():
    g = build(); n = g["n"]; Y = g["Y"]; B = g["B"]; u_0 = g["u_0"]; words = g["words"]; a0 = g["a0"]
    graph = {"n": n, "A": g["A"], "D": g["D"]}
    US.N = n; US.BATCH = n; US.T_MAX = 8000; US.BURN = 1000; US.MIN_T = 3000; US.CHUNK = 1000; US.STOP_RHAT = 1.05
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    inits = [np.full(n, -0.5), np.full(n, 0.5), np.full(n, 1.5), np.random.RandomState(7000).randn(n) * 1.5]
    ctx = dict(Y=Y, B=B, u_0=u_0, BtB=BtB, P=P, Lc=Lc, a0=a0, inits=inits, seed_base=0)
    print(f"[Amazon 리뷰] n={n} 후보단어, 관측 seed={[w for w in SEED_WORDS if w in list(words)]} (Y=1 {int(Y.sum())}개)\n", flush=True)

    def metrics(states):
        posts, wts = US.collect(states); pn = np.zeros(n); pd = 0.0; ess = []; minU = []; statU = []
        for ci in range(4):
            post = posts[ci]
            if wts[ci] is not None:
                sw = wts[ci].sum(); pn += (wts[ci][:, None] * post).sum(0); pd += sw
            else:
                pn += post.sum(0); pd += post.shape[0]
            ess.append(float(np.nanmedian(US.ess_per_node(post))))
            Utr = E.energy_trace_common(post, Y, B, u_0, a0); minU.append(float(Utr.min())); statU.append(float(np.median(Utr)))
        th = pn / pd; R = US.split_rhat_coords(posts, wts)
        return dict(rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)), rmax=float(np.nanmax(R)),
                    ess=float(np.mean(ess)), minU=minU, statU=statU, th=th)

    D = {}
    aw, T_conv = US.run_method("AWSGLD", ctx, target=None)
    D["AWSGLD"] = metrics(aw); cutoff = float(np.round(np.median(D["AWSGLD"]["statU"]))); del aw
    for m in ["SGLD", "qSGLD", "cycSGLD", "SGHMC"]:
        sts, _ = US.run_method(m, ctx, target=T_conv); D[m] = metrics(sts); del sts
    E.T = T_conv; E.BURN = US.BURN; E.BATCH = US.BATCH
    posts = []
    for ci, ini in enumerate(inits):
        np.random.seed(ci); posts.append(dict(method="acMH", ths=E.run_acmh(graph, Y, B, u_0, ini, a0, ci), t=T_conv))
    D["acMH"] = metrics(posts); del posts

    order_m = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
    print(f"cutoff U={cutoff:.0f}, T_conv={T_conv}\n[표1 수렴·효율]")
    print(f"{'Sampler':>8} | {'R̂med':>6} {'R̂q95':>6} {'R̂max':>6} | {'ESS':>5} | {'LowU':>6}")
    for m in order_m:
        d = D[m]; low = float(np.mean([max(cutoff, v) for v in d["minU"]]))
        print(f"{m:>8} | {d['rmed']:>6.3f} {d['rq95']:>6.3f} {d['rmax']:>6.3f} | {d['ess']:>5.0f} | {low:>6.0f}")
    print("\n[top-12 키워드 (π̂=sigmoid(θ̂) 내림차순)]  * = 관측 seed")
    for m in order_m:
        th = D[m]["th"]; top = np.argsort(th)[::-1][:12]
        toks = [(words[i] + ("*" if Y[i] == 1 else "")) for i in top]
        print(f"  {m:>8}: {', '.join(toks)}")
    print("\n[논문 BSS(Table 1.8)]: mccann, oatmeal*, lovers/loved, personally, every, like, well, known, alone")


if __name__ == "__main__":
    main()
