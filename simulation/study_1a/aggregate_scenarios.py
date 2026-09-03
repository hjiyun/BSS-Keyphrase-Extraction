"""unified_scenarios_s*.csv (시드별) → 10시드 mean±std 2표 출력 + 집계 CSV.
표1(수렴·추정): R̂med/q95/max · ESS · θ̂(S/W/N) · π̂(S/W/N) · Lowest U
표2(순위): Spearman · Kendall · Top-k · NDCG@50 · MSE_all
"""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SCEN_ORDER = ["Easy", "Moderate", "Difficult", "Sparse"]
METH_ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "th_S", "th_W", "th_N",
       "pi_S", "pi_W", "pi_N", "lowest_U", "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"]
MU = {"Easy": (2.5, 1.0, -2.5), "Moderate": (2.0, 0.5, -1.8),
      "Difficult": (1.5, 0.0, -1.0), "Sparse": (2.0, 1.0, -1.0)}


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


rows = []
for f in sorted(glob.glob(os.path.join(_HERE, "unified_scenarios_s*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
seeds = sorted(set(int(r["seed"]) for r in rows))
print(f"집계: {len(rows)} 행, 시드 {seeds}\n")

agg = {}
for scen in SCEN_ORDER:
    for meth in METH_ORDER:
        vals = {mt: [] for mt in MET}
        for r in rows:
            if r["scenario"] == scen and r["method"] == meth:
                for mt in MET:
                    vals[mt].append(float(r[mt]))
        agg[(scen, meth)] = {mt: (np.mean(v), np.std(v), len(v)) for mt, v in vals.items() if v}


def c(scen, meth, mt, fmt="{:.3f}"):
    m, s, _ = agg[(scen, meth)][mt]
    return f"{fmt.format(m)}±{fmt.format(s)}"


def trio(scen, meth, keys, fmt="{:.2f}"):
    return "/".join(fmt.format(agg[(scen, meth)][k][0]) for k in keys)


for scen in SCEN_ORDER:
    ns = agg[(scen, "AWSGLD")]["rhat_median"][2]
    tconv = agg[(scen, "AWSGLD")]["T_stop"][0]
    mu = MU[scen]; pistar = tuple(sig(np.array(mu)))
    print(f"\n########## {scen}  (n_seed={ns}, T_conv≈{tconv:.0f}) ##########")
    print(f"  정답 θ*(S/W/N) ≈ {mu[0]:.1f}/{mu[1]:.1f}/{mu[2]:.1f}   π*(S/W/N) ≈ {pistar[0]:.2f}/{pistar[1]:.2f}/{pistar[2]:.2f}")
    print("\n[표1 수렴·추정]  R̂/ESS/LowU=mean±std, θ̂·π̂=그룹평균(10시드 평균)")
    print(f"{'Sampler':>8} | {'R̂med':>11} {'R̂q95':>11} {'R̂max':>11} | {'ESS':>10} | {'θ̂(S/W/N)':>16} {'π̂(S/W/N)':>16} | {'LowU':>9}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {c(scen,meth,'rhat_median'):>11} {c(scen,meth,'rhat_q95'):>11} {c(scen,meth,'rhat_max'):>11} | "
              f"{c(scen,meth,'ess','{:.1f}'):>10} | {trio(scen,meth,['th_S','th_W','th_N']):>16} "
              f"{trio(scen,meth,['pi_S','pi_W','pi_N']):>16} | {c(scen,meth,'lowest_U','{:.0f}'):>9}")
    print("\n[표2 순위]  mean±std")
    print(f"{'Sampler':>8} | {'Spear':>11} {'Kendall':>11} {'Top-k':>13} {'NDCG@50':>13} {'MSE_all':>13}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {c(scen,meth,'spearman','{:.2f}'):>11} {c(scen,meth,'kendall','{:.2f}'):>11} "
              f"{c(scen,meth,'topk','{:.3f}'):>13} {c(scen,meth,'ndcg50','{:.3f}'):>13} {c(scen,meth,'mse_all','{:.2f}'):>13}")

with open(os.path.join(_HERE, "unified_scenarios_agg.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    hdr = ["scenario", "method", "n_seed"]
    for mt in MET:
        hdr += [f"{mt}_mean", f"{mt}_std"]
    w.writerow(hdr)
    for scen in SCEN_ORDER:
        for meth in METH_ORDER:
            row = [scen, meth, agg[(scen, meth)]["rhat_median"][2]]
            for mt in MET:
                m, s, _ = agg[(scen, meth)][mt]
                row += [round(m, 5), round(s, 5)]
            w.writerow(row)
print("\n저장: unified_scenarios_agg.csv")
