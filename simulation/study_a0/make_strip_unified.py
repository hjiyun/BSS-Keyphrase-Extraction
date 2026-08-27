"""unified_detail_n100.csv → strip (Lowest U 클램프 + ESS). 4연쇄×seed0, 연쇄별 점."""
import os, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_F):
    fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
ORDER = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}

low = {}; ess = {}; CUT = None; NC = 0
for r in csv.DictReader(open(os.path.join(HERE, "unified_detail_n100.csv"))):
    m = r["method"]; low.setdefault(m, []).append(float(r["lowest_U_clamped"]))
    ess.setdefault(m, []).append(float(r["ess"])); CUT = float(r["cutoff"])
METHODS = [m for m in ORDER if m in low]; NC = len(low["AWSGLD"])

fig, axes = plt.subplots(1, 2, figsize=(15, 5.2)); rng = np.random.RandomState(0)
for ax, dic, ylab, ttl, fmt, hl in [
        (axes[0], low, "Lowest $U$ reached (clamped at cut-off)", "(a) Lowest U reached  (lower = better)", "{:.0f}", CUT),
        (axes[1], ess, "ESS (median over nodes)", "(b) ESS  (higher = better)", "{:.1f}", None)]:
    for i, m in enumerate(METHODS):
        v = np.array(dic[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
        ax.scatter(xs, v, s=52, color=COL[m], alpha=0.85, edgecolor="white", lw=0.6, zorder=3)
        mean = v.mean(); ax.plot([i - 0.28, i + 0.28], [mean, mean], color=COL[m], lw=3, zorder=4)
        ax.text(i, v.max(), fmt.format(mean), ha="center", va="bottom", fontsize=10, color=COL[m], fontweight="bold")
    if hl is not None:
        ax.axhline(hl, color="#C0392B", ls="--", lw=1.2, alpha=0.7, label=f"cut-off U={hl:.0f}")
        ax.legend(fontsize=9, loc="upper left")
    ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=9)
    for lbl, m in zip(ax.get_xticklabels(), METHODS):
        if m == "AWSGLD": lbl.set_color(COL[m]); lbl.set_fontweight("bold")
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11, fontweight="bold"); ax.grid(True, axis="y", alpha=0.25)
fig.suptitle(f"BSS posterior (n=100, T=20000) — Lowest U (clamped at cut-off U={CUT:.0f}) + ESS  ({len(METHODS)} samplers, {NC}-chain × seed0)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "strip6_cutoff_n100.png"), dpi=140, bbox_inches="tight")
print(f"저장: strip6_cutoff_n100.png (methods {METHODS}, {NC}-chain, cutoff {CUT:.0f})")
