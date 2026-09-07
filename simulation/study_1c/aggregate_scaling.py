"""unified_scaling_n*_s*.csv (크기·시드별) → 크기별 5시드 mean±std 2표 + 집계 CSV.
표1(수렴·추정): R̂med/q95/max · ESS · θ̂(S/W/N) · π̂(S/W/N) · Lowest U
표2(순위): Spearman · Kendall · Top-k · NDCG@50 · MSE_all
"""
import os, glob, csv
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = [200, 1500, 10000]
METH_ORDER = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
MET = ["rhat_median", "rhat_q95", "rhat_max", "ess", "th_S", "th_W", "th_N",
       "pi_S", "pi_W", "pi_N", "lowest_U", "spearman", "kendall", "topk", "ndcg50", "mse_all", "T_stop"]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def mu_of(n):
    try:
        d = np.load(os.path.join(_HERE, f"data_n{n}_seed0.npz"))
        return (float(d["param_mu_S"]), float(d["param_mu_W"]), float(d["param_mu_N"]))
    except Exception:
        return (None, None, None)


rows = []
for f in sorted(glob.glob(os.path.join(_HERE, "unified_scaling_n*_s*.csv"))):
    for r in csv.DictReader(open(f)):
        rows.append(r)
seeds = sorted(set(int(r["seed"]) for r in rows))
present = sorted(set(int(r["n"]) for r in rows))
print(f"집계: {len(rows)} 행, 크기 {present}, 시드 {seeds}\n")

agg = {}
for n in present:
    for meth in METH_ORDER:
        vals = {mt: [] for mt in MET}
        for r in rows:
            if int(r["n"]) == n and r["method"] == meth:
                for mt in MET:
                    vals[mt].append(float(r[mt]))
        if vals["rhat_median"]:
            agg[(n, meth)] = {mt: (np.mean(v), np.std(v), len(v)) for mt, v in vals.items()}


def c(n, meth, mt, fmt="{:.3f}"):
    m, s, _ = agg[(n, meth)][mt]
    return f"{fmt.format(m)}±{fmt.format(s)}"


def trio(n, meth, keys, fmt="{:.2f}"):
    return "/".join(fmt.format(agg[(n, meth)][k][0]) for k in keys)


for n in present:
    if (n, "AWSGLD") not in agg:
        continue
    ns = agg[(n, "AWSGLD")]["rhat_median"][2]; tconv = agg[(n, "AWSGLD")]["T_stop"][0]
    mu = mu_of(n); ps = tuple(sig(np.array([x if x is not None else 0 for x in mu])))
    print(f"\n########## n={n}  (n_seed={ns}, T_conv≈{tconv:.0f}) ##########")
    if mu[0] is not None:
        print(f"  정답 θ*(S/W/N) ≈ {mu[0]:.1f}/{mu[1]:.1f}/{mu[2]:.1f}   π*(S/W/N) ≈ {ps[0]:.2f}/{ps[1]:.2f}/{ps[2]:.2f}")
    print("\n[표1 수렴·추정]  R̂/ESS/LowU=mean±std, θ̂·π̂=그룹평균(시드 평균)")
    print(f"{'Sampler':>8} | {'R̂med':>11} {'R̂q95':>11} {'R̂max':>11} | {'ESS':>10} | {'θ̂(S/W/N)':>16} {'π̂(S/W/N)':>16} | {'LowU':>9}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {c(n,meth,'rhat_median'):>11} {c(n,meth,'rhat_q95'):>11} {c(n,meth,'rhat_max'):>11} | "
              f"{c(n,meth,'ess','{:.1f}'):>10} | {trio(n,meth,['th_S','th_W','th_N']):>16} "
              f"{trio(n,meth,['pi_S','pi_W','pi_N']):>16} | {c(n,meth,'lowest_U','{:.0f}'):>9}")
    print("\n[표2 순위]  mean±std")
    print(f"{'Sampler':>8} | {'Spear':>11} {'Kendall':>11} {'Top-k':>13} {'NDCG@50':>13} {'MSE_all':>13}")
    for meth in METH_ORDER:
        print(f"{meth:>8} | {c(n,meth,'spearman','{:.2f}'):>11} {c(n,meth,'kendall','{:.2f}'):>11} "
              f"{c(n,meth,'topk','{:.3f}'):>13} {c(n,meth,'ndcg50','{:.3f}'):>13} {c(n,meth,'mse_all','{:.2f}'):>13}")

with open(os.path.join(_HERE, "unified_scaling_agg.csv"), "w", newline="") as fh:
    w = csv.writer(fh)
    hdr = ["n", "method", "n_seed"]
    for mt in MET:
        hdr += [f"{mt}_mean", f"{mt}_std"]
    w.writerow(hdr)
    for n in present:
        for meth in METH_ORDER:
            if (n, meth) not in agg:
                continue
            row = [n, meth, agg[(n, meth)]["rhat_median"][2]]
            for mt in MET:
                m, s, _ = agg[(n, meth)][mt]
                row += [round(m, 5), round(s, 5)]
            w.writerow(row)
print("\n저장: unified_scaling_agg.csv")
