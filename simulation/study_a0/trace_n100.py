"""AWSGLD vs SGLD 사후 trace — 통합 setup(4-chain×seed0)의 chain 0 (bad init μ_N).
AWSGLD 는 inline run_awsgld(=통합표와 동일 경로), SGLD 는 E.run_sgld_family. 둘 다 seed 0, μ_N.
대표 노드(S/W/N 각 2)의 θ_t. 점선=정답 θ*.
출력: trace_n100.png (합본), trace_n100_AWSGLD.png, trace_n100_SGLD.png
"""
import os, sys, time
import numpy as np
from scipy.stats import invgamma
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
import data_generator as DG
from local_trap_landscape import PARAMS

_F = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(_F):
    fm.fontManager.addfont(_F); plt.rcParams["font.family"] = fm.FontProperties(fname=_F).get_name()
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100; E.T = 20000; E.BURN = 2000; E.BATCH = 50
T = E.T; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
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


def run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    """통합표와 동일한 inline AWSGLD."""
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 40)); es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        eps = 0.3 / ((t + 1) ** 0.6 + 10)
        U = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2)
        bidx = np.random.choice(n, E.BATCH, replace=False) if E.BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, alpha, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        gm = 1.0
        if t < warm:
            es.append(U)
            if t == warm - 1:
                lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
        else:
            J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
            gm = float(np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths


def main():
    graph, Y, B, u_0, ts, z = gen(N, 0); n = N
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
    a0 = E.alpha_find(u_0, Y, E.GRID); ini = np.full(n, float(E.MU_N))
    rng = np.random.RandomState(1); picks = []
    for g in ("S", "W", "N"):
        idx = np.where(z == g)[0]; picks += list(rng.choice(idx, 2, replace=False))
    t0 = time.time()
    print(f"trace n={N} T={T} | AWSGLD(inline, chain0) vs SGLD | bad init μ_N | seed 0", flush=True)
    th_aw = run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed=0)
    print(f"  AWSGLD done ({int(time.time()-t0)}s)", flush=True)
    np.random.seed(0); th_sg = E.run_sgld_family("SGLD", graph, Y, B, u_0, ini, a0, 0)
    print(f"  SGLD done ({int(time.time()-t0)}s)", flush=True)

    from matplotlib.lines import Line2D
    leg = [Line2D([0], [0], color=GCOL[g], lw=2, label=f"{g} (θ*≈{PARAMS['mu_'+g]:+.1f})") for g in ("S", "W", "N")]

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

    for ths, nm in [(th_aw, "AWSGLD"), (th_sg, "SGLD")]:
        fig, ax = plt.subplots(figsize=(13, 4.6))
        for i in picks:
            g = z[i]; ax.plot(ths[:, i], color=GCOL[g], lw=0.6, alpha=0.8)
            ax.axhline(ts[i], color=GCOL[g], ls=":", lw=1.0, alpha=0.6)
        ax.axvline(E.BURN, color="k", ls="--", lw=1, alpha=0.4)
        ax.axhline(float(E.MU_N), color="red", ls="-", lw=0.8, alpha=0.3)
        ax.set_ylabel(r"$\theta_t$"); ax.set_xlabel("iteration"); ax.grid(alpha=0.15)
        ax.legend(handles=leg, fontsize=9, loc="upper right")
        ax.set_title(f"{nm}  (n={N}, T={E.T};  chain0 from bad init μ_N={E.MU_N};  dotted = true θ*)",
                     fontsize=11, fontweight="bold")
        fig.tight_layout(); fig.savefig(os.path.join(HERE, f"trace_n100_{nm}.png"), dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"저장: trace_n100_{nm}.png", flush=True)
    print(f"({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
