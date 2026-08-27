"""AWSGLD vs SGLD 사후 trace (θ 궤적) — n=100, T=20000, bad init μ_N.
대표 노드(S/W/N 각 2개)의 θ_t 를 반복에 따라 그림. 점선 = 정답 θ*.
AWSGLD 는 bad init 에서 정답 영역으로 올라가 안정 진동(수렴), SGLD 는 헤맴.
출력: trace_n100.png
"""
import os, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_diagnostics as E
import data_generator as DG
from local_trap_landscape import PARAMS

_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_F):
    fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100; E.T = 20000; E.BURN = 2000; E.BATCH = 50
GCOL = {"S": "#2F6DB2", "W": "#D85A30", "N": "#6B6B6B"}


def gen(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, ts, z


def main():
    graph, Y, B, u_0, ts, z = gen(N, 0); n = N
    a0 = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
    rng = np.random.RandomState(1); picks = []
    for g in ("S", "W", "N"):
        idx = np.where(z == g)[0]; picks += list(rng.choice(idx, 2, replace=False))
    t0 = time.time()
    print(f"trace n={N} T={E.T} | AWSGLD, SGLD | bad init μ_N", flush=True)
    th_aw = E.run_awsgld(graph, Y, B, u_0, ini, a0, 0); print(f"  AWSGLD done ({int(time.time()-t0)}s)", flush=True)
    th_sg = E.run_sgld_family("SGLD", graph, Y, B, u_0, ini, a0, 0); print(f"  SGLD done ({int(time.time()-t0)}s)", flush=True)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=GCOL[g], lw=2, label=f"{g} (θ*≈{PARAMS['mu_'+g]:+.1f})") for g in ("S", "W", "N")]

    # ── 합본 (2행) ──
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for ax, ths, nm in [(axes[0], th_aw, "AWSGLD"), (axes[1], th_sg, "SGLD")]:
        for i in picks:
            g = z[i]; ax.plot(ths[:, i], color=GCOL[g], lw=0.6, alpha=0.8)
            ax.axhline(ts[i], color=GCOL[g], ls=":", lw=1.0, alpha=0.6)
        ax.axvline(E.BURN, color="k", ls="--", lw=1, alpha=0.4)
        ax.axhline(float(E.MU_N), color="red", ls="-", lw=0.8, alpha=0.3)
        ax.set_ylabel(f"{nm}\n" + r"$\theta_t$"); ax.grid(alpha=0.15)
        ax.set_title(f"{nm}  (from bad init μ_N={E.MU_N};  dotted = true θ*,  S blue / W orange / N gray)",
                     fontsize=10, fontweight="bold")
    axes[0].legend(handles=leg, fontsize=9, loc="upper right"); axes[1].set_xlabel("iteration")
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "trace_n100.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
    print("저장: trace_n100.png (합본)", flush=True)

    # ── 각 1장 ──
    for ths, nm in [(th_aw, "AWSGLD"), (th_sg, "SGLD")]:
        fig, ax = plt.subplots(figsize=(13, 4.6))
        for i in picks:
            g = z[i]
            ax.plot(ths[:, i], color=GCOL[g], lw=0.6, alpha=0.8)
            ax.axhline(ts[i], color=GCOL[g], ls=":", lw=1.0, alpha=0.6)
        ax.axvline(E.BURN, color="k", ls="--", lw=1, alpha=0.4)
        ax.axhline(float(E.MU_N), color="red", ls="-", lw=0.8, alpha=0.3)
        ax.set_ylabel(r"$\theta_t$"); ax.set_xlabel("iteration"); ax.grid(alpha=0.15)
        ax.legend(handles=leg, fontsize=9, loc="upper right")
        ax.set_title(f"{nm}  (n={N}, T={E.T};  from bad init μ_N={E.MU_N};  dotted = true θ*)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout()
        out = os.path.join(HERE, f"trace_n100_{nm}.png")
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"저장: trace_n100_{nm}.png", flush=True)
    print(f"({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
