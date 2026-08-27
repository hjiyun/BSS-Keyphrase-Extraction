"""full_metrics_n100.csv → 6패널 막대그림 (strip6_cutoff.png 스타일, AWSGLD 강조).
AWSGLD=π가중, 나머지=raw. R̂ 패널엔 1.2 수렴선."""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_FONT):
    fm.fontManager.addfont(_FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}

d = {}
for r in csv.DictReader(open(os.path.join(HERE, "full_metrics_n100.csv"))):
    d[r["method"]] = r
METHODS = [m for m in ORDER if m in d]   # CSV 에 있는 것만 (acMH 병합되면 자동 포함)
V = lambda m, k: float(d[m][k])

panels = [
    ("Spearman (↑ better)", "spearman", None, "{:.2f}", False),
    ("MSE (↓ better)", "mse", None, "{:.2f}", False),
    ("NDCG@50 (↑ better)", "ndcg50", None, "{:.3f}", False),
    ("Rhat max (↓, 1.05/1.2 선)", "rhat_max", 1.05, "{:.2f}", True),
    ("Rhat median (↓ better)", "rhat_median", 1.05, "{:.3f}", True),
    ("ESS (↑ better)", "ess", None, "{:.0f}", False),
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for ax, (title, key, hline, fmt, isr) in zip(axes.flat, panels):
    vals = [V(m, key) for m in METHODS]
    bars = ax.bar(range(len(METHODS)), vals, color=[COL[m] for m in METHODS], alpha=0.88)
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=25, ha="right", fontsize=9)
    for lbl, m in zip(ax.get_xticklabels(), METHODS):
        if m == "AWSGLD": lbl.set_color(COL[m]); lbl.set_fontweight("bold")
    ax.set_title(title, fontsize=11, fontweight="bold"); ax.grid(axis="y", alpha=0.2)
    if hline is not None:
        ax.axhline(1.05, color="red", ls="--", lw=1.1, alpha=0.75)
        ax.axhline(1.2, color="orange", ls=":", lw=1.1, alpha=0.6)
    for i, m in enumerate(METHODS):
        v = vals[i]
        ax.text(i, v + (0.01 if v >= 0 else -0.01) * (max(vals) - min(vals) + 1),
                fmt.format(v), ha="center", va="bottom" if v >= 0 else "top",
                fontsize=8.5, fontweight="bold" if m == "AWSGLD" else "normal",
                color=COL[m] if m == "AWSGLD" else "black")
fig.suptitle(f"BSS posterior (n=100, T=20000) — {len(METHODS)} samplers, all metrics  |  AWSGLD = π-weighted (정직), others = raw",
             fontsize=13, fontweight="bold")
fig.tight_layout()
out = os.path.join(HERE, "full_metrics_n100.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print("저장:", out)
