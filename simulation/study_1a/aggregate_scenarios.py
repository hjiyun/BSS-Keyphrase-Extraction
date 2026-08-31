"""unified_scenarios_s*.csv (시드별) → 10시드 mean±std 2표 출력 + 집계 CSV."""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SCEN_ORDER = ["Easy", "Moderate", "Difficult", "Sparse"]
METH_ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "mse_theta", "mse_pi",
       "spearman", "kendall", "topk", "ndcg50", "T_stop"]

rows = []
for f in sorted(glob.glob(os.path.join(_HERE, "unified_scenarios_s*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
seeds = sorted(set(int(r["seed"]) for r in rows))
print(f"집계: {len(rows)} 행, 시드 {seeds}\n")

agg = {}  # (scen, meth) -> {metric: (mean,std,n)}
for scen in SCEN_ORDER:
    for meth in METH_ORDER:
        vals = {mt: [] for mt in MET}
        for r in rows:
            if r["scenario"] == scen and r["method"] == meth:
                for mt in MET:
                    vals[mt].append(float(r[mt]))
        agg[(scen, meth)] = {mt: (np.mean(v), np.std(v), len(v)) for mt, v in vals.items() if v}


def cell(scen, meth, mt, fmt="{:.3f}"):
    m, s, _ = agg[(scen, meth)][mt]
    return f"{fmt.format(m)}±{fmt.format(s)}"


for scen in SCEN_ORDER:
    nseed = agg[(scen, "AWSGLD")]["rhat_median"][2]
    tconv = agg[(scen, "AWSGLD")]["T_stop"][0]
    print(f"\n########## {scen}  (n_seed={nseed}, AWSGLD T_conv≈{tconv:.0f}) ##########")
    print("\n[표1 수렴]  mean±std")
    print(f"{'Sampler':>8} | {'R̂med':>13} {'R̂q95':>13} {'R̂max':>13} | {'ESS':>11} | {'MSE_θ':>13} {'MSE_π':>15}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {cell(scen,meth,'rhat_median'):>13} {cell(scen,meth,'rhat_q95'):>13} "
              f"{cell(scen,meth,'rhat_max'):>13} | {cell(scen,meth,'ess','{:.1f}'):>11} | "
              f"{cell(scen,meth,'mse_theta','{:.2f}'):>13} {cell(scen,meth,'mse_pi','{:.4f}'):>15}")
    print("\n[표2 순위]  mean±std")
    print(f"{'Sampler':>8} | {'Spear':>13} {'Kendall':>13} {'Top-k':>13} {'NDCG@50':>13} {'MSE_all':>13}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {cell(scen,meth,'spearman','{:.2f}'):>13} {cell(scen,meth,'kendall','{:.2f}'):>13} "
              f"{cell(scen,meth,'topk','{:.3f}'):>13} {cell(scen,meth,'ndcg50','{:.3f}'):>13} "
              f"{cell(scen,meth,'mse_theta','{:.2f}'):>13}")

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
