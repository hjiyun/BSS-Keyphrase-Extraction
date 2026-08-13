"""6개 샘플러 strip plot — cyc_awsgld.png 와 X·Y축 동일 (Y='Lowest U reached'), 단 cutoff 에서 멈춤.

cyc_awsgld.png(Rastrigin)과 완전히 같은 형식:
  X = 6 samplers, Y = Lowest U reached (낮을수록 좋음), seed 별 점 + 평균 막대.
차이: 에너지가 cut-off(U≤CUT)에 '도달하면 거기서 멈춤'.
  → 잘 내려가는 샘플러는 전부 cut-off 선에 모여 '비슷'해지고(AWSGLD·cycSGLD·acMH),
     못 내려가는 샘플러(qSGLD·SGLD·SGHMC)는 그 위에 뜬다.
  → lowest 축에서 AWSGLD≈cycSGLD, 진짜 차이는 ESS 패널.

cut-off=312 (수렴 지표 계산값: R̂max<1.2 수렴 후 정상상태 에너지, AWSGLD 3seed 평균 312.2±11.9).
샘플러 구현은 원본 Study 1B(energy_diagnostics)와 100% 동일.
출력: strip6_cutoff.png / .csv
"""
import os, sys, time, csv
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_SEEDS = [0, 1, 2, 3, 4]
# cutoff = 수렴 지표가 계산한 값: running R̂max<1.2 수렴 후 정상상태 에너지 (AWSGLD 3seed 평균 312.2±11.9)
CUT = 312
BURN = E.BURN
METHODS = ["acMH", "SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]
COL = {"acMH": "#E1B12C", "SGLD": "#95a5a6", "qSGLD": "#27ae60",
       "cycSGLD": "#8e44ad", "SGHMC": "#16A085", "AWSGLD": "#2456A6"}


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"]


def lowest_clamped(U, cut):
    """cut-off 아래로 내려가면 전부 cut-off 로 클램프. 도달자=cut, 미도달자=실제 최저(>cut)."""
    return float(max(cut, U.min()))


def main():
    lowU = {m: [] for m in METHODS}; ess = {m: [] for m in METHODS}
    t0 = time.time()
    print(f"6-sampler strip | Lowest U reached, stop@U≤{CUT} | seeds={DATA_SEEDS}\n", flush=True)
    for s in DATA_SEEDS:
        graph, Y, B, u_0 = load(s); n = graph["n"]
        a_star = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
        for m in METHODS:
            ths = E.RUNNERS[m](graph, Y, B, u_0, ini, a_star, 1000 + s)
            U = E.energy_trace_common(ths, Y, B, u_0, a_star)
            lowU[m].append(lowest_clamped(U, CUT))
            ess[m].append(float(np.nanmedian(ess_per_node(ths[BURN:]))))
        print(f"  seed {s}: " + "  ".join(f"{m}={lowU[m][-1]:.0f}" for m in METHODS) + f"  ({int(time.time()-t0)}s)", flush=True)

    print("\n=== 평균 (Lowest U reached, 낮을수록 좋음 / ESS 높을수록 좋음) ===")
    for m in METHODS:
        reached = sum(1 for x in lowU[m] if x <= CUT + 1)
        print(f"  {m:>8}: lowestU={np.mean(lowU[m]):>4.0f}  (cut-off 도달 {reached}/{len(DATA_SEEDS)})  ESS={np.mean(ess[m]):>5.1f}")

    with open(os.path.join(HERE, "strip6_cutoff.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "seed", "lowest_U_stop", "ess"])
        for m in METHODS:
            for i, s in enumerate(DATA_SEEDS):
                w.writerow([m, s, round(lowU[m][i], 1), round(ess[m][i], 2)])

    # ── strip plot (cyc_awsgld.png 형식 그대로): 점 + 평균 막대 + 값 라벨 ──
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
    fig.suptitle(f"BSS posterior — Lowest U reached, clamped at data-derived cut-off U={CUT} (R̂-converged energy) — reachers tie, ESS favors AWSGLD",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "strip6_cutoff.png"), dpi=140, bbox_inches="tight")
    print(f"\n저장: strip6_cutoff.png, strip6_cutoff.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
