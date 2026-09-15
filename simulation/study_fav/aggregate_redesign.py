"""redesign_*.csv → config별 확정 3표: 표1 FDR@γ / 표2 순위 / 표3 σ̂·Jaccard·cov90."""
import os, glob, csv, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
METH = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
GAMMAS = [0.05, 0.1, 0.15, 0.2, 0.3]
pat = sys.argv[1] if len(sys.argv) > 1 else "redesign_*.csv"

for fn in sorted(glob.glob(os.path.join(_HERE, pat))):
    raw = list(csv.reader(open(fn))); meta = raw[0]; hdr = raw[1]; rows = [dict(zip(hdr, r)) for r in raw[2:]]
    seeds = sorted(set(int(r["seed"]) for r in rows))
    def ms(m, k, f="{:.3f}"):
        v = [float(r[k]) for r in rows if r["method"] == m and r.get(k, "") not in ("", "nan")]
        return (f.format(np.mean(v)) + "±" + f.format(np.std(v))) if v else "-"
    tr, n, mu, al = meta[1], meta[3], meta[5], meta[7]
    print(f"\n{'#'*72}\n조건: n={n}  μ±{mu}  α={al}  (truth≈{tr}, {len(seeds)}시드)  θ*={mu}/0/-{mu}\n{'#'*72}")
    print("[표1 γ별 FDR 제어]  precision / recall / F1 / realFDR")
    for g in GAMMAS:
        print(f"  γ={g}")
        print(f"  {'method':>8} | {'precision':>13} {'recall':>13} {'F1':>13} {'realFDR':>13}")
        for m in METH:
            print(f"  {m:>8} | {ms(m,f'P{g}'):>13} {ms(m,f'R{g}'):>13} {ms(m,f'F{g}'):>13} {ms(m,f'rFDR{g}'):>13}")
    print("\n[표2 순위]  Spearman / Kendall / Top-k / NDCG / MSE / AUC")
    print(f"  {'method':>8} | {'Spearman':>11} {'Kendall':>11} {'Top-k':>11} {'NDCG':>11} {'MSE':>11} {'AUC':>11}")
    for m in METH:
        print(f"  {m:>8} | {ms(m,'spearman'):>11} {ms(m,'kendall'):>11} {ms(m,'topk'):>11} {ms(m,'ndcg'):>11} {ms(m,'mse','{:.2f}'):>11} {ms(m,'auc'):>11}")
    print("\n[표3 σ̂(사후폭) / Jaccard(재현성) / cov90(90%CI 커버리지)]")
    print(f"  {'method':>8} | {'sigma':>13} {'Jaccard':>13} {'cov90':>13}")
    for m in METH:
        print(f"  {m:>8} | {ms(m,'sigma_raw','{:.2f}'):>13} {ms(m,'jaccard'):>13} {ms(m,'cov90'):>13}")
