"""문서 병합(고중첩 3편·10편) 통합표 — study_1c 와 동일 지표 세트(실데이터 적응형).

실데이터라 연속 θ* 는 없고 binary 키워드 gold(truth)만 있다. 따라서:
  표1(수렴·추정): R̂med/q95/max · ESS · θ̂(kw/non-kw) · π̂(kw/non-kw) · Lowest U
  표2(순위, gold 기준): P@k · R@k · NDCG@k(binary) · AUC   (k=|truth|)
6 샘플러(acMH 포함), 4 연쇄, AWSGLD R̂median<1.05 조기종료(T_MAX 상한), 5 시드(Y 관측 시드).
병합은 merge_multimodal 방식(공유 어휘, 문서 내 동시출현만). acMH=원본 componentwise.

사용: python3 merge_unified.py <ndoc> <seed> [tag]   # ndoc∈{3,10}
"""
import os, sys, csv, time
import numpy as np
from numpy.linalg import solve
from scipy.stats import invgamma

_HERE = os.path.dirname(os.path.abspath(__file__))
_1C = os.path.join(os.path.dirname(_HERE), "study_1c")
_A0 = os.path.join(os.path.dirname(_HERE), "study_a0")
ROOT = os.path.dirname(os.path.dirname(_HERE)); CODE = os.path.join(ROOT, "code_JOC")
for p in (_1C, _A0, os.path.join(_A0, "_archive"), CODE, _HERE):
    sys.path.insert(0, p)
import importlib.util
import keyphrase_functions_awsgld as kfa                       # noqa: E402
from keyphrase_functions_awsgld import alpha_find              # noqa: E402
import energy_diagnostics as E                                 # acMH·alpha_find·energy_trace  # noqa: E402
import unified_scaling as US                                   # 연속 샘플러 재사용  # noqa: E402

os.chdir(_HERE)  # exp.build_graph 가 상대경로 사용
spec = importlib.util.spec_from_file_location("exp", os.path.join(_HERE, "acmh_vs_awsgld_4to10.py"))
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
d_damp = 0.85; grid = exp.grid; OBS_RATIO = 0.20

import csv as _csv
dense = [r[0] for r in _csv.reader(open(os.path.join(exp.BASE, "selected_ids.txt")))]
G_ALL = {}
for dc in dense:
    try:
        G_ALL[dc] = exp.build_graph(dc)
    except Exception:
        pass
IDS = list(G_ALL); V = {k: set(G_ALL[k]["words"]) for k in IDS}


_RANK = None
_DISTINCT = {}


def _greedy(start, ndoc):
    sel = [start]
    while len(sel) < ndoc:
        cand = max((x for x in IDS if x not in sel), key=lambda x: sum(len(V[x] & V[s]) for s in sel))
        sel.append(cand)
    return sel


def high_overlap_variant(ndoc, variant, k=10):
    """서로 다른(집합 기준 distinct) 고중첩 세트 k개 중 variant번째.
    총중첩 순위대로 시작문서를 바꿔 탐욕 생성하되, 집합이 겹치면 건너뛰어 distinct k개 확보."""
    global _RANK
    if ndoc not in _DISTINCT:
        if _RANK is None:
            _RANK = sorted(IDS, key=lambda x: sum(len(V[x] & V[s]) for s in IDS if s != x), reverse=True)
        sets = []; seen = set()
        for start in _RANK:
            s = _greedy(start, ndoc); fs = frozenset(s)
            if fs not in seen:
                seen.add(fs); sets.append(s)
            if len(sets) >= k:
                break
        _DISTINCT[ndoc] = sets
    return _DISTINCT[ndoc][variant]


def merge(docs, yseed):
    vocab = sorted(set().union(*[V[x] for x in docs]))
    idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab)
    A = np.zeros((n, n))
    for x in docs:
        g = G_ALL[x]; wmap = [idx[w] for w in g["words"]]
        A[np.ix_(wmap, wmap)] += g["A"]
    np.fill_diagonal(A, 0); deg = A.sum(1); keep = deg > 0
    if not keep.all():
        A = A[np.ix_(keep, keep)]; vocab = [w for w, k in zip(vocab, keep) if k]
        idx = {w: i for i, w in enumerate(vocab)}; n = len(vocab); deg = A.sum(1)
    D = np.diag(deg); Bm = np.eye(n) - d_damp * solve(D, A).T
    u_0 = solve(Bm, np.ones(n) * (1 - d_damp))
    truth_doc = {}
    for x in docs:
        g = G_ALL[x]
        truth_doc[x] = sorted({idx[g["words"][t]] for t in g["truth"] if g["words"][t] in idx})
    truth = sorted(set().union(*truth_doc.values()))
    rng = np.random.RandomState(20260721 + yseed)
    Y = np.zeros(n)
    for x in docs:
        td = truth_doc[x]
        if td:
            k = max(1, round(OBS_RATIO * len(td)))
            Y[list(rng.choice(td, min(k, len(td)), replace=False))] = 1
    a0 = alpha_find(u_0, Y, grid)
    return dict(n=n, A=A, D=D, B=Bm, u_0=u_0, Y=Y.astype(float), truth=np.array(truth),
                a0=a0, docs=docs)


# ── gold 기반 순위 지표 ──
def gold_metrics(th, truth, n):
    order = np.argsort(th)[::-1]
    tset = set(truth.tolist()); k = len(truth)
    sel = order[:k]; tp = sum(1 for i in sel if i in tset)
    P = tp / k if k else 0.0; R = tp / k if k else 0.0
    # NDCG@k (binary rel)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    rel = np.array([1.0 if i in tset else 0.0 for i in order[:k]])
    dcg = float(np.sum(rel * disc)); idcg = float(np.sum(disc[:min(k, len(truth))]))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    # ROC AUC (rank 기반): score=th, label=in truth
    pos = k; neg = n - k
    if pos == 0 or neg == 0:
        auc = float("nan")
    else:
        ranks = np.empty(n); ranks[np.argsort(th)] = np.arange(1, n + 1)
        sum_pos = float(np.sum(ranks[list(tset)]))
        auc = (sum_pos - pos * (pos + 1) / 2) / (pos * neg)
    return P, R, ndcg, auc


def run_one(ndoc, variant):
    seed = variant                                    # Y 관측 시드 = 변형 번호(세트마다 고정)
    mg = merge(high_overlap_variant(ndoc, variant), seed)
    n = mg["n"]; Y = mg["Y"]; B = mg["B"]; u_0 = mg["u_0"]; truth = mg["truth"]; a0 = mg["a0"]
    graph = {"n": n, "A": mg["A"], "D": mg["D"]}
    US.N = n; US.BATCH = n; US.T_MAX = 8000; US.BURN = 1000; US.MIN_T = 3000; US.CHUNK = 1000; US.STOP_RHAT = 1.05
    t0 = time.time()
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    inits = [np.full(n, -0.5), np.full(n, 0.5), np.full(n, 1.5),
             np.random.RandomState(7000 + seed).randn(n) * 1.5]
    ctx = dict(Y=Y, B=B, u_0=u_0, BtB=BtB, P=P, Lc=Lc, a0=a0, inits=inits, seed_base=100 * seed)
    print(f"[ndoc={ndoc} seed={seed}] merged n={n}, truth={len(truth)}, docs={mg['docs']} precond({int(time.time()-t0)}s)", flush=True)

    # AWSGLD 먼저 → T_conv + cutoff
    aw_states, T_conv = US.run_method("AWSGLD", ctx, target=None)
    statU = [float(np.median(E.energy_trace_common(aw_states[c]["ths"][US.BURN:aw_states[c]["t"]], Y, B, u_0, a0)))
             for c in range(len(aw_states))]
    cutoff = float(np.round(np.median(statU)))
    D = {"AWSGLD": _metrics(aw_states, ctx, truth, cutoff, T_conv)}
    del aw_states
    print(f"    AWSGLD T_conv={T_conv} cutoff={cutoff:.0f} ({int(time.time()-t0)}s)", flush=True)

    for m in ["SGLD", "qSGLD", "cycSGLD", "SGHMC"]:
        sts, _ = US.run_method(m, ctx, target=T_conv)
        D[m] = _metrics(sts, ctx, truth, cutoff, T_conv); del sts
        print(f"    {m:>8} ({int(time.time()-t0)}s)", flush=True)
    # acMH: 원본 componentwise, T_conv 까지
    E.T = T_conv; E.BURN = US.BURN; E.BATCH = US.BATCH
    posts = []
    for ci, ini in enumerate(inits):
        np.random.seed(100 * seed + ci)
        ths = E.run_acmh(graph, Y, B, u_0, ini, a0, 100 * seed + ci)
        posts.append(dict(method="acMH", ths=ths, t=T_conv))
    D["acMH"] = _metrics(posts, ctx, truth, cutoff, T_conv); del posts
    print(f"    acMH ({int(time.time()-t0)}s)", flush=True)

    rows = []
    for m in ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]:
        d = D[m]
        rows.append([ndoc, seed, n, len(truth), m, round(d["rmed"], 4), round(d["rq95"], 4), round(d["rmax"], 4),
                     round(d["ess"], 2), round(d["thK"], 4), round(d["thN"], 4), round(d["piK"], 4), round(d["piN"], 4),
                     round(d["low"], 2), round(d["P"], 4), round(d["R"], 4), round(d["ndcg"], 4),
                     round(d["auc"], 4), d["Tstop"]])
    return rows


def _metrics(states, ctx, truth, cutoff, T_conv):
    Y, B, u_0, a0 = ctx["Y"], ctx["B"], ctx["u_0"], ctx["a0"]; n = US.N
    posts, wts = US.collect(states)
    pn = np.zeros(n); pd = 0.0; ess_list = []; minU = []
    for ci in range(len(states)):
        post = posts[ci]
        if wts[ci] is not None:
            sw = wts[ci].sum(); pn += (wts[ci][:, None] * post).sum(0); pd += sw
        else:
            pn += post.sum(0); pd += post.shape[0]
        ess_list.append(float(np.nanmedian(US.ess_per_node(post))))
        minU.append(float(E.energy_trace_common(post, Y, B, u_0, a0).min()))
    th = pn / pd; R = US.split_rhat_coords(posts, wts)
    pih = 1.0 / (1.0 + np.exp(-np.clip(th, -700, 700)))
    kwmask = np.zeros(n, bool); kwmask[truth] = True
    P, Rc, ndcg, auc = gold_metrics(th, truth, n)
    return dict(rmed=float(np.median(R)), rq95=float(np.quantile(R, 0.95)), rmax=float(np.nanmax(R)),
                ess=float(np.mean(ess_list)),
                thK=float(th[kwmask].mean()), thN=float(th[~kwmask].mean()),
                piK=float(pih[kwmask].mean()), piN=float(pih[~kwmask].mean()),
                low=float(np.mean([max(cutoff, v) for v in minU])),
                P=P, R=Rc, ndcg=ndcg, auc=auc, Tstop=T_conv)


def main():
    ndoc = int(sys.argv[1]); seed = int(sys.argv[2])
    tag = sys.argv[3] if len(sys.argv) > 3 else f"d{ndoc}_s{seed}"
    rows = run_one(ndoc, seed)
    out = os.path.join(_HERE, f"merge_unified_{tag}.csv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ndoc", "seed", "n", "n_truth", "method", "rhat_median", "rhat_q95", "rhat_max", "ess",
                    "th_kw", "th_nonkw", "pi_kw", "pi_nonkw", "lowest_U", "P_at_k", "R_at_k", "ndcg_at_k", "auc", "T_stop"])
        w.writerows(rows)
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
