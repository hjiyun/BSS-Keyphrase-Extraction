"""merge_unified_d*_s*.csv → 병합조건(3편·10편)별 5시드 mean±std 2표 + 집계 CSV.
표1(수렴·추정): R̂med/q95/max · ESS · θ̂(kw/non) · π̂(kw/non) · Lowest U
표2(gold 순위): P@k · R@k · NDCG@k · AUC   (k=|truth|)
"""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
METH_ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "th_kw", "th_nonkw", "pi_kw", "pi_nonkw",
       "lowest_U", "P_at_k", "R_at_k", "ndcg_at_k", "auc", "T_stop", "n", "n_truth"]

rows = []
for f in sorted(glob.glob(os.path.join(_HERE, "merge_unified_d*_s*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
seeds = sorted(set(int(r["seed"]) for r in rows))
conds = sorted(set(int(r["ndoc"]) for r in rows))
print(f"집계: {len(rows)} 행, 조건 {conds}편, 시드 {seeds}\n")

agg = {}
for nd in conds:
    for meth in METH_ORDER:
        vals = {mt: [] for mt in MET}
        for r in rows:
            if int(r["ndoc"]) == nd and r["method"] == meth:
                for mt in MET:
                    vals[mt].append(float(r[mt]))
        if vals["rhat_median"]:
            agg[(nd, meth)] = {mt: (np.mean(v), np.std(v)) for mt, v in vals.items()}


def c(nd, meth, mt, fmt="{:.3f}"):
    m, s = agg[(nd, meth)][mt]
    return f"{fmt.format(m)}±{fmt.format(s)}"


def duo(nd, meth, keys, fmt="{:.2f}"):
    return "/".join(fmt.format(agg[(nd, meth)][k][0]) for k in keys)


for nd in conds:
    if (nd, "AWSGLD") not in agg:
        continue
    nn = agg[(nd, "AWSGLD")]["n"][0]; nt = agg[(nd, "AWSGLD")]["n_truth"][0]; tc = agg[(nd, "AWSGLD")]["T_stop"][0]
    print(f"\n########## 병합 {nd}편  (n={nn:.0f}, truth={nt:.0f}, T={tc:.0f}) ##########")
    print("\n[표1 수렴·추정]  R̂/ESS/LowU=mean±std, θ̂·π̂=kw/non-kw 그룹평균")
    print(f"{'Sampler':>8} | {'R̂med':>11} {'R̂q95':>11} {'R̂max':>11} | {'ESS':>10} | {'θ̂(kw/non)':>13} {'π̂(kw/non)':>13} | {'LowU':>9}")
    for meth in METH_ORDER:
        if (nd, meth) not in agg:
            continue
        print(f"{meth:>8} | {c(nd,meth,'rhat_median'):>11} {c(nd,meth,'rhat_q95'):>11} {c(nd,meth,'rhat_max'):>11} | "
              f"{c(nd,meth,'ess','{:.1f}'):>10} | {duo(nd,meth,['th_kw','th_nonkw']):>13} "
              f"{duo(nd,meth,['pi_kw','pi_nonkw']):>13} | {c(nd,meth,'lowest_U','{:.0f}'):>9}")
    print("\n[표2 gold 순위]  mean±std  (k=|truth|)")
    print(f"{'Sampler':>8} | {'P@k':>11} {'R@k':>11} {'NDCG@k':>11} {'AUC':>11}")
    for meth in METH_ORDER:
        if (nd, meth) not in agg:
            continue
        print(f"{meth:>8} | {c(nd,meth,'P_at_k'):>11} {c(nd,meth,'R_at_k'):>11} {c(nd,meth,'ndcg_at_k'):>11} {c(nd,meth,'auc'):>11}")

with open(os.path.join(_HERE, "merge_unified_agg.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    hdr = ["ndoc", "method"]
    for mt in MET:
        hdr += [f"{mt}_mean", f"{mt}_std"]
    w.writerow(hdr)
    for nd in conds:
        for meth in METH_ORDER:
            if (nd, meth) not in agg:
                continue
            row = [nd, meth]
            for mt in MET:
                m, s = agg[(nd, meth)][mt]
                row += [round(m, 5), round(s, 5)]
            w.writerow(row)
print("\n저장: merge_unified_agg.csv")
