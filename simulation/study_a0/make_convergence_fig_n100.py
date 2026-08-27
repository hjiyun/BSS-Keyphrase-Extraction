"""awsgld_convergence_n100.csv → R̂max 수렴곡선 (수렴선 1.05). AWSGLD=가중 R̂."""
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
CK = [2000, 5000, 10000, 15000, 20000]
COL = {"AWSGLD": "#2456A6", "qSGLD": "#27ae60", "cycSGLD": "#8e44ad", "SGHMC": "#16A085"}
rows = {r["method"]: r for r in csv.DictReader(open(os.path.join(HERE, "awsgld_convergence_n100.csv")))}

fig, ax = plt.subplots(figsize=(9, 5.4))
for m, c in COL.items():
    ys = [float(rows[m][f"rhatmax@{k}"]) for k in CK]
    ax.plot(CK, ys, "o-", color=c, lw=1.9, ms=6, label=m + (" (π가중)" if m == "AWSGLD" else ""))
ax.axhline(1.05, color="red", ls="--", lw=1.3, alpha=0.8, label="R̂=1.05 (엄격 수렴 기준)")
ax.axhline(1.2, color="orange", ls=":", lw=1.1, alpha=0.6, label="R̂=1.2 (완화 기준)")
# AWSGLD 1.05 통과 지점 표시
aw = [float(rows["AWSGLD"][f"rhatmax@{k}"]) for k in CK]
ax.annotate("AWSGLD: 15k에서 1.03\n(1.05 기준 수렴)", (15000, aw[3]),
            xytext=(9000, 1.9), fontsize=9.5, color=COL["AWSGLD"], fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=COL["AWSGLD"]))
ax.set_xlabel("iteration (T)"); ax.set_ylabel("R̂ max (π 기준)")
ax.set_title("수렴 속도 (n=100): AWSGLD만 R̂max<1.05 도달 (과분산 3-chain)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9); ax.grid(alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "convergence_n100_r105.png"), dpi=140, bbox_inches="tight")
print("저장: convergence_n100_r105.png")
