"""strip6_cutoff.png 와 동일 형식 — n=100 기준.
(a) Lowest U reached (cutoff 클램프)  (b) ESS.  6 샘플러, seed별 점 + 평균 막대.
cutoff = AWSGLD 정상상태(수렴 영역) 에너지 median (seed 평균). 데이터는 n=100 새로 생성.
"""
import os, sys, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import data_generator as DG
from local_trap_landscape import PARAMS
from extra_metrics import ess_per_node

_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_F):
    fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100; E.T = 20000; E.BURN = 2000; E.BATCH = 50
BURN = E.BURN
DATA_SEEDS = [0, 1, 2, 3, 4]
METHODS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}


def gen(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0


def main():
    minU = {m: [] for m in METHODS}; ess = {m: [] for m in METHODS}; statE = {m: [] for m in METHODS}
    t0 = time.time()
    print(f"strip6 (n={N}) | 6 sampler × {len(DATA_SEEDS)} seed | T={E.T}\n", flush=True)
    for s in DATA_SEEDS:
        graph, Y, B, u_0 = gen(N, s); n = N
        a_star = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
        for m in METHODS:
            ths = E.RUNNERS[m](graph, Y, B, u_0, ini, a_star, 1000 + s)
            U = E.energy_trace_common(ths, Y, B, u_0, a_star)
            minU[m].append(float(U[BURN:].min())); statE[m].append(float(np.median(U[BURN:])))
            ess[m].append(float(np.nanmedian(ess_per_node(ths[BURN:]))))
        print(f"  seed {s} done ({int(time.time()-t0)}s)", flush=True)

    CUT = float(np.round(np.mean(statE["AWSGLD"])))   # cutoff = AWSGLD 정상상태 에너지 (수렴 영역)
    lowU = {m: [max(CUT, v) for v in minU[m]] for m in METHODS}   # cutoff 아래는 클램프
    print(f"\ncutoff = AWSGLD 정상상태 에너지 평균 = {CUT:.0f}")
    for m in METHODS:
        reached = sum(1 for v in lowU[m] if v <= CUT + 1e-6)
        print(f"  {m:>8}: lowestU(클램프) {np.mean(lowU[m]):>5.0f} (도달 {reached}/{len(DATA_SEEDS)})  ESS {np.mean(ess[m]):>5.1f}")

    with open(os.path.join(HERE, "strip6_cutoff_n100.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "lowest_U_clamped", "min_U", "ess", "cutoff"])
        for m in METHODS:
            for i, s in enumerate(DATA_SEEDS):
                w.writerow([m, s, round(lowU[m][i], 1), round(minU[m][i], 1), round(ess[m][i], 2), CUT])

    # ── strip: (a) Lowest U (클램프) (b) ESS ── (strip6_cutoff.png 형식)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2)); rng = np.random.RandomState(0)
    for ax, dic, ylab, ttl, fmt, hl in [
            (axes[0], lowU, "Lowest $U$ reached (clamped at cut-off)", "(a) Lowest U reached  (lower = better)", "{:.0f}", CUT),
            (axes[1], ess, "ESS (median over nodes)", "(b) ESS  (higher = better)", "{:.1f}", None)]:
        for i, m in enumerate(METHODS):
            v = np.array(dic[m]); xs = i + (rng.rand(len(v)) - 0.5) * 0.28
            ax.scatter(xs, v, s=48, color=COL[m], alpha=0.8, edgecolor="white", lw=0.6, zorder=3)
            mean = v.mean(); ax.plot([i - 0.28, i + 0.28], [mean, mean], color=COL[m], lw=3, zorder=4)
            ax.text(i, v.max(), fmt.format(mean), ha="center", va="bottom", fontsize=10, color=COL[m], fontweight="bold")
        if hl is not None:
            ax.axhline(hl, color="#C0392B", ls="--", lw=1.2, alpha=0.7, label=f"cut-off U={hl:.0f}")
            ax.legend(fontsize=9, loc="upper left")
        ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=20, ha="right", fontsize=9)
        for lbl, m in zip(ax.get_xticklabels(), METHODS):
            if m == "AWSGLD": lbl.set_color(COL[m]); lbl.set_fontweight("bold")
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=11, fontweight="bold"); ax.grid(True, axis="y", alpha=0.25)
    fig.suptitle(f"BSS posterior (n={N}) — Lowest U reached (clamped at cut-off U={CUT:.0f}) + ESS  ({len(DATA_SEEDS)} seeds, bad init μ_N)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strip6_cutoff_n100.png"), dpi=140, bbox_inches="tight")
    print(f"\n저장: strip6_cutoff_n100.png/.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
