"""AWSGLD 복원·샘플링 품질 종합 그래프 — 기본 BSS 에너지 U(θ) (Study 1B, mixture 아님).

3개 그림:
  1. recovery_scatter.png    — θ̂ vs θ* 산점도 (6 sampler). 대각선=완벽 복원.
  2. sampling_quality_bars.png — Spearman / MSE / R̂max / ESS median 막대 (6 sampler).
  3. convergence_diag.png     — running MSE(θ̄_k vs θ*) + 대표 노드 trace.

전부 Study 1B 저장 체인(ava_results.npz, sgld_results.npz) = 실제 BSS 사후분포 샘플링 결과.
"""
import os, sys, json
import numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
S1B = os.path.join(HERE, "..", "study_1b")   # 대용량 체인(ava/sgld_results.npz)·요약은 study_1b 참조
ava = np.load(os.path.join(S1B, "ava_results.npz"), allow_pickle=True)
sg = np.load(os.path.join(S1B, "sgld_results.npz"), allow_pickle=True)
theta_star = ava["theta_star"]; z = np.array([str(x) for x in ava["z"]]); BURN = int(ava["BURN_IN"])
avaS = json.load(open(os.path.join(S1B, "ava_metric_summary.json")))
sgS = json.load(open(os.path.join(S1B, "sgld_metric_summary.json")))

NAMES = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}
SRC = {"acMH": ava, "AWSGLD": ava, "SGLD": sg, "qSGLD": sg, "cycSGLD": sg, "SGHMC": sg}
GCOL = {"S": "#2F6DB2", "W": "#D85A30", "N": "#6B6B6B"}

def theta_hat(m): return SRC[m][f"{m}_theta_hat"]
def rhat_max(m):
    j = avaS if m in ("acMH", "AWSGLD") else sgS
    return j["R_hat"][m]["R_hat_max"]
def recov(m):
    j = avaS if m in ("acMH", "AWSGLD") else sgS
    return j["recovery"][m]["spearman"], j["recovery"][m]["mse_all"]
def ess_med(m):
    ch = SRC[m][f"{m}_chain0_theta_store"][BURN:]
    return float(np.nanmedian(ess_per_node(ch)))

# ── 1. 복원 산점도 ──
fig, axes = plt.subplots(2, 3, figsize=(13, 8.5))
lim = (theta_star.min() - 0.5, theta_star.max() + 0.5)
for ax, m in zip(axes.flat, NAMES):
    th = theta_hat(m); sp, mse = recov(m)
    for g in ("N", "W", "S"):
        ax.scatter(theta_star[z == g], th[z == g], s=10, color=GCOL[g], alpha=0.5, label=g)
    ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_title(f"{m}   (Spearman {sp:.3f}, MSE {mse:.2f})", fontsize=10,
                 fontweight="bold" if m == "AWSGLD" else "normal", color=COL[m])
    ax.set_xlabel("true $\\theta^*$"); ax.set_ylabel("estimated $\\hat\\theta$"); ax.grid(alpha=0.15)
    if m == "acMH": ax.legend(fontsize=8, markerscale=1.5, loc="upper left")
fig.suptitle("Recovery on the BSS posterior $U(\\theta)$: estimated vs true (diagonal = perfect)",
             fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "recovery_scatter.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("저장: recovery_scatter.png")

# ── 2. 샘플링 품질 4-막대 ──
spearman = {m: recov(m)[0] for m in NAMES}
mse = {m: recov(m)[1] for m in NAMES}
rhat = {m: rhat_max(m) for m in NAMES}
ess = {m: ess_med(m) for m in NAMES}
fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
panels = [("Spearman (↑ better)", spearman, False, 1.1),
          ("MSE (↓ better)", mse, False, None),
          ("$\\hat R$ max (↓, <1.2 converged)", rhat, True, None),
          ("ESS median (↑ better)", ess, False, None)]
for ax, (title, dic, hline, top) in zip(axes, panels):
    cols = [COL[m] for m in NAMES]
    ax.bar(range(len(NAMES)), [dic[m] for m in NAMES], color=cols, alpha=0.85)
    ax.set_xticks(range(len(NAMES))); ax.set_xticklabels(NAMES, rotation=30, ha="right", fontsize=8)
    for lbl, m in zip(ax.get_xticklabels(), NAMES):
        if m == "AWSGLD": lbl.set_fontweight("bold"); lbl.set_color(COL[m])
    ax.set_title(title, fontsize=10, fontweight="bold"); ax.grid(axis="y", alpha=0.2)
    if hline: ax.axhline(1.2, color="red", ls="--", lw=1, alpha=0.6)
    if top: ax.set_ylim(0, top)
    for i, m in enumerate(NAMES):
        ax.text(i, dic[m], f"{dic[m]:.2f}", ha="center", va="bottom", fontsize=7.5)
fig.suptitle("Sampling quality on the BSS posterior $U(\\theta)$ (Study 1B, n=400)", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "sampling_quality_bars.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("저장: sampling_quality_bars.png")

# ── 3. 수렴 진단: running MSE + trace ──
def running_mse(m):
    ts = SRC[m][f"{m}_chain0_theta_store"]
    cum = np.cumsum(ts, axis=0) / np.arange(1, ts.shape[0] + 1)[:, None]
    return np.mean((cum - theta_star[None, :]) ** 2, axis=1)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ax = axes[0]
for m in NAMES:
    ax.plot(running_mse(m), color=COL[m], lw=1.6 if m == "AWSGLD" else 1.0,
            alpha=0.9 if m == "AWSGLD" else 0.7, label=m)
ax.axvline(BURN, color="k", ls=":", lw=1, alpha=0.5)
ax.set_yscale("log"); ax.set_xlabel("iteration"); ax.set_ylabel("running MSE  $\\|\\bar\\theta_k-\\theta^*\\|^2/n$")
ax.set_title("(a) running posterior-mean MSE vs $\\theta^*$", fontsize=11, fontweight="bold")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2, which="both")
# (b) AWSGLD 대표 노드 trace (S/W/N 각 2개)
ax = axes[1]; ts = ava["AWSGLD_chain0_theta_store"]
rngp = np.random.RandomState(1)
for g in ("S", "W", "N"):
    idx = np.where(z == g)[0]; pick = rngp.choice(idx, 2, replace=False)
    for i in pick:
        ax.plot(ts[:, i], color=GCOL[g], lw=0.6, alpha=0.8)
        ax.axhline(theta_star[i], color=GCOL[g], ls=":", lw=0.8, alpha=0.5)
ax.axvline(BURN, color="k", ls=":", lw=1, alpha=0.5)
ax.set_xlabel("iteration"); ax.set_ylabel("$\\theta_k^{(i)}$")
ax.set_title("(b) AWSGLD component traces (dotted = true $\\theta^*$)", fontsize=11, fontweight="bold")
ax.grid(alpha=0.2)
fig.suptitle("Convergence diagnostics on the BSS posterior $U(\\theta)$", fontsize=12, fontweight="bold")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "convergence_diag.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
print("저장: convergence_diag.png")

print("\n=== 요약 (BSS 사후분포 복원·샘플링) ===")
print(f"{'sampler':>8} {'Spearman':>9} {'MSE':>6} {'R̂max':>7} {'ESS':>7}")
for m in NAMES:
    print(f"{m:>8} {spearman[m]:>9.3f} {mse[m]:>6.2f} {rhat[m]:>7.2f} {ess[m]:>7.1f}")
