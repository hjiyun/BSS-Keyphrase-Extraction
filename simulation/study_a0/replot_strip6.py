"""strip6_cutoff.csv 재활용: cut-off 아래로 내려간 값은 전부 cut-off 에 클램프해 다시 그림.
(재실행 없음) 도달한 샘플러는 정확히 cut-off 선에 모이고, 못 내려간 샘플러만 위에 뜬다.
"""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CUT = 312
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}

lowU = {m: [] for m in METHODS}; ess = {m: [] for m in METHODS}
with open(os.path.join(HERE, "strip6_cutoff.csv")) as fh:
    for row in csv.DictReader(fh):
        m = row["method"]
        lowU[m].append(max(CUT, float(row["lowest_U_stop"])))   # cut-off 아래는 cut-off 로 클램프
        ess[m].append(float(row["ess"]))

print(f"클램프 후 평균 (Lowest U, cut-off={CUT} / ESS)")
for m in METHODS:
    reached = sum(1 for v in lowU[m] if v <= CUT + 1e-6)
    print(f"  {m:>8}: lowestU={np.mean(lowU[m]):>4.0f} (도달 {reached}/{len(lowU[m])})  ESS={np.mean(ess[m]):>5.1f}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
rng = np.random.RandomState(0)
for ax, dic, ylab, ttl, fmt, hline in [
        (axes[0], lowU, "Lowest $U$ reached (clamped at cut-off)",
         "(a) Lowest U reached  (lower = better)", "{:.0f}", CUT),
        (axes[1], ess, "ESS (median over nodes)", "(b) ESS  (higher = better)", "{:.1f}", None)]:
    for i, m in enumerate(METHODS):
        v = np.array(dic[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
        ax.scatter(xs, v, s=48, color=COL[m], alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
        mean = v.mean()
        ax.plot([i - 0.28, i + 0.28], [mean, mean], color=COL[m], lw=3, zorder=4)
        ax.text(i, v.max(), fmt.format(mean), ha="center", va="bottom",
                fontsize=10, color=COL[m], fontweight="bold")
    if hline is not None:
        ax.axhline(hline, color="#C0392B", ls="--", lw=1.2, alpha=0.7, label=f"cut-off U={hline}")
        ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=9)
    for lbl, m in zip(ax.get_xticklabels(), METHODS):
        if m == "AWSGLD": lbl.set_color(COL[m]); lbl.set_fontweight("bold")
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
fig.suptitle(f"BSS posterior — Lowest U reached, clamped at data-derived cut-off U=312 (Rhat<1.2 converged energy) — reachers tie, ESS favors AWSGLD",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "strip6_cutoff.png"), dpi=140, bbox_inches="tight")
print("저장: strip6_cutoff.png (클램프 버전)")
