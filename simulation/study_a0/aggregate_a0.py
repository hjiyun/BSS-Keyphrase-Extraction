"""unified_a0_s*.csv (시드별) → A0 10시드 mean±std 2표 + 집계 CSV.
표1(수렴·추정): R̂med/q95/max·ESS·θ̂(S/W/N)·π̂(S/W/N)·LowU
표2(순위): Spearman·Kendall·Top-k·NDCG@50·MSE_all
"""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
METH = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "th_S", "th_W", "th_N",
       "pi_S", "pi_W", "pi_N", "lowest_U", "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"]
MU = (2.5, 1.0, -0.8)

rows = []
for f in sorted(glob.glob(os.path.join(_HERE, "unified_a0_s*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
seeds = sorted(set(int(r["seed"]) for r in rows))
print(f"집계: {len(rows)} 행, 시드 {seeds}")

agg = {}
for m in METH:
    v = {mt: [float(r[mt]) for r in rows if r["method"] == m] for mt in MET}
    agg[m] = {mt: (np.mean(x), np.std(x)) for mt, x in v.items() if x}


def c(m, mt, fmt="{:.3f}"):
    a, s = agg[m][mt]; return f"{fmt.format(a)}±{fmt.format(s)}"


def trio(m, keys, fmt="{:.2f}"):
    return "/".join(fmt.format(agg[m][k][0]) for k in keys)


tc = agg["AWSGLD"]["T_stop"][0]
ps = tuple(1/(1+np.exp(-np.array(MU))))
print(f"\n########## Study A0 (원본 U, n=100)  10시드  T_conv≈{tc:.0f} ##########")
print(f"  정답 θ*(S/W/N) ≈ {MU[0]}/{MU[1]}/{MU[2]}   π*(S/W/N) ≈ {ps[0]:.2f}/{ps[1]:.2f}/{ps[2]:.2f}")
print("\n[표1 수렴·추정]  R̂/ESS/LowU=mean±std, θ̂·π̂=그룹평균")
print(f"{'Sampler':>8} | {'R̂med':>11} {'R̂q95':>11} {'R̂max':>11} | {'ESS':>10} | {'θ̂(S/W/N)':>16} {'π̂(S/W/N)':>16} | {'LowU':>9}")
for m in METH:
    print(f"{m:>8} | {c(m,'rhat_median'):>11} {c(m,'rhat_q95'):>11} {c(m,'rhat_max'):>11} | {c(m,'ess','{:.1f}'):>10} | "
          f"{trio(m,['th_S','th_W','th_N']):>16} {trio(m,['pi_S','pi_W','pi_N']):>16} | {c(m,'lowest_U','{:.0f}'):>9}")
print("\n[표2 순위]  mean±std")
print(f"{'Sampler':>8} | {'Spear':>11} {'Kendall':>11} {'Top-k':>13} {'NDCG@50':>13} {'MSE_all':>13}")
for m in METH:
    print(f"{m:>8} | {c(m,'spearman','{:.2f}'):>11} {c(m,'kendall','{:.2f}'):>11} {c(m,'topk','{:.3f}'):>13} "
          f"{c(m,'ndcg50','{:.3f}'):>13} {c(m,'mse_all','{:.2f}'):>13}")

with open(os.path.join(_HERE, "unified_a0_agg.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["method", "n_seed"] + [f"{mt}_{s}" for mt in MET for s in ("mean", "std")])
    for m in METH:
        row = [m, len(seeds)]
        for mt in MET:
            a, s = agg[m][mt]; row += [round(a, 5), round(s, 5)]
        w.writerow(row)
print("\n저장: unified_a0_agg.csv")
