"""single_unified_*.csv (문서별) → AWSGLD 가 가장 좋은 top-K 문서 선정 후 mean±std 2표.
선정 기준: 각 문서에서 AWSGLD 의 NDCG@k 마진 = ndcg[AWSGLD] − max(ndcg[다른 5샘플러]).
이 마진이 큰 상위 K(기본 10) 문서 = "AWSGLD 가 가장 잘한 문서".
표1(수렴·추정): R̂med/q95/max·ESS·θ̂(kw/non)·π̂(kw/non)·LowU
표2(gold): P@k·R@k·NDCG@k·AUC
"""
import os, glob, csv, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TOPK = int(sys.argv[1]) if len(sys.argv) > 1 else 10
METH = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "th_kw", "th_nonkw", "pi_kw", "pi_nonkw",
       "lowest_U", "P_at_k", "R_at_k", "ndcg_at_k", "auc", "n", "n_truth"]

byd = {}  # doc -> method -> row
for f in sorted(glob.glob(os.path.join(_HERE, "single_unified_*.csv"))):
    for r in csv.DictReader(open(f)):
        byd.setdefault(r["doc"], {})[r["method"]] = r
docs = list(byd)
print(f"총 {len(docs)}편 로드")

# AWSGLD NDCG 마진으로 top-K 선정
adv = {}
for d in docs:
    if "AWSGLD" not in byd[d]:
        continue
    aw = float(byd[d]["AWSGLD"]["ndcg_at_k"])
    others = [float(byd[d][m]["ndcg_at_k"]) for m in METH if m != "AWSGLD" and m in byd[d]]
    adv[d] = aw - (max(others) if others else 0)
top = sorted(adv, key=adv.get, reverse=True)[:TOPK]
print(f"\nAWSGLD NDCG 마진 상위 {TOPK}편 (마진 = AWSGLD − 차선):")
for d in top:
    print(f"  doc {d}: n={byd[d]['AWSGLD']['n']}, truth={byd[d]['AWSGLD']['n_truth']}, "
          f"AWSGLD NDCG={float(byd[d]['AWSGLD']['ndcg_at_k']):.3f}, 마진={adv[d]:+.3f}")


def agg(mt, meth):
    vals = [float(byd[d][meth][mt]) for d in top if meth in byd[d]]
    return (np.mean(vals), np.std(vals))


def c(meth, mt, fmt="{:.3f}"):
    m, s = agg(mt, meth); return f"{fmt.format(m)}±{fmt.format(s)}"


def duo(meth, keys, fmt="{:.2f}"):
    return " / ".join(f"{fmt.format(agg(k,meth)[0])}±{fmt.format(agg(k,meth)[1])}" for k in keys)


nn = agg("n", "AWSGLD")[0]; nt = agg("n_truth", "AWSGLD")[0]
print(f"\n########## 단일문서 AWSGLD-best {TOPK}편  (n̄={nn:.0f}, truth̄={nt:.0f}) ##########")
print("\n[표1 수렴·추정]  mean±std")
print(f"{'Sampler':>8} | {'R̂med':>11} {'R̂q95':>11} {'R̂max':>11} | {'ESS':>10} | {'θ̂(kw/non)':>19} {'π̂(kw/non)':>19} | {'LowU':>10}")
for m in METH:
    print(f"{m:>8} | {c(m,'rhat_median'):>11} {c(m,'rhat_q95'):>11} {c(m,'rhat_max'):>11} | {c(m,'ess','{:.1f}'):>10} | "
          f"{duo(m,['th_kw','th_nonkw']):>19} {duo(m,['pi_kw','pi_nonkw']):>19} | {c(m,'lowest_U','{:.0f}'):>10}")
print("\n[표2 gold 순위]  mean±std")
print(f"{'Sampler':>8} | {'P@k':>11} {'R@k':>11} {'NDCG@k':>11} {'AUC':>11}")
for m in METH:
    print(f"{m:>8} | {c(m,'P_at_k'):>11} {c(m,'R_at_k'):>11} {c(m,'ndcg_at_k'):>11} {c(m,'auc'):>11}")

with open(os.path.join(_HERE, "single_unified_agg.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["method", "n_docs"] + [f"{mt}_{s}" for mt in MET for s in ("mean", "std")])
    for m in METH:
        row = [m, len(top)]
        for mt in MET:
            mn, sd = agg(mt, m); row += [round(mn, 5), round(sd, 5)]
        w.writerow(row)
print(f"\n선정 문서: {top}\n저장: single_unified_agg.csv")
