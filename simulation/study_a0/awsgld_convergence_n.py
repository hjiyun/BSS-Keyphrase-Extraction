"""수렴 속도 (작은 n) — n을 줄여 조건수를 낮추고, 각 샘플러가 R̂<1.2에 실제 도달하는지.

n=400은 조건수가 나빠 20k에도 아무도 R̂max<1.2를 못 넘었다. n을 낮추면(예: 100) 수렴이
도달 가능해진다. AWSGLD(가중 R̂)가 실제로 수렴하고 cyc/qSGLD보다 빠른지 확인.
데이터는 Study 1B와 동일 PARAMS·구성으로 n만 바꿔 새로 생성(기존 n=400 데이터 불변).

인자: python3 awsgld_convergence_n.py [N]   (기본 100)
출력: awsgld_convergence_n{N}.png / .csv
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
import data_generator as DG
from local_trap_landscape import PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100
T_MAX = 20000; BURN = 500; BATCH = min(50, N); FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
CYC = 10; SEED = 0
INITS = [-0.8, 1.0, 2.5, "rand"]  # 4-chain
CKPTS = [2000, 5000, 10000, 15000, 20000]
METHODS = ["AWSGLD", "qSGLD", "cycSGLD", "SGHMC"]
COL = {"AWSGLD": "#2456A6", "qSGLD": "#27ae60", "cycSGLD": "#8e44ad", "SGHMC": "#16A085"}


def gen_data(n, seed):
    """Study 1B 구성으로 n만 바꿔 생성."""
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    theta_star = DG.sample_theta_star(z, PARAMS, rng)
    Y_clean, _ = DG.sample_Y(theta_star, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Y_clean, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, theta_star


def run(method, ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T_MAX, n)); Js = np.full(T_MAX, -1); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T_MAX // 40)); es = []; emin = du = None; J = M_REG - 1
    use_aw = method == "AWSGLD"
    for t in range(T_MAX):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        if method == "cycSGLD":
            cl = max(1, T_MAX // CYC); beta = (t % cl) / cl
            eps = 0.02 / 2.0 * (np.cos(np.pi * min(beta, 0.8)) + 1); tau_k = TAU if beta >= 0.8 else TAU / 1e4
        else:
            eps = 0.3 / ((t + 1) ** 0.6 + 10); tau_k = TAU
        U = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2)
        bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
        gU = kfa.grad_posterior_energy(Y, alpha, theta, u_0, B, s2, batch_idx=bidx, BtB=BtB)
        gm = 1.0
        if use_aw:
            if t < warm:
                es.append(U)
                if t == warm - 1:
                    lo, hi = min(es), max(es); rg = max(hi - lo, 1.0)
                    emin = lo - 0.5 * rg; du = max((hi + 0.5 * rg - emin) / M_REG, 1e-8); es = None
            else:
                J = int(np.clip((U - emin) / du + 1, 1, M_REG - 1))
                gm = float(np.clip(1 + (ZETA * tau_k / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        if use_aw:
            theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * tau_k * eps) * (Lc @ np.random.randn(n))
            if t >= warm:
                dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
                aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        elif method == "qSGLD":
            theta = theta - eps * (P @ gU) + np.sqrt(2 * tau_k * eps) * (Lc @ np.random.randn(n))
        else:  # cycSGLD / SGHMC(간이)
            theta = theta - eps * gU + np.sqrt(2 * tau_k * eps) * np.random.randn(n)
        theta = np.clip(theta, -700, 700)
        ths[t] = theta; Js[t] = J; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths, Js, aw


def rhat_w(chains, Jchains, aw, lo, hi, weighted):
    """split-R̂: [lo:hi] 를 반으로 갈라 2M sub-chain. AWSGLD 는 π-가중."""
    means = []; withins = []; L = hi - lo; h = L // 2; hL = h
    for ci, c in enumerate(chains):
        for a, b in ((lo, lo + h), (lo + h, lo + 2 * h)):
            post = c[a:b]
            if weighted:
                Jp = Jchains[ci][a:b]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
                m = (w[:, None] * post).sum(0) / sw; v = (w[:, None] * (post - m) ** 2).sum(0) / sw
            else:
                m = post.mean(0); v = post.var(0, ddof=1)
            means.append(m); withins.append(v)
    means = np.array(means); withins = np.array(withins); M2 = len(means)
    gm = means.mean(0); Bo = ((means - gm) ** 2).sum(0) / (M2 - 1); W = withins.mean(0)
    R = np.sqrt(np.clip(((hL - 1) / hL * W + Bo) / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R)), float(np.median(R))


def main():
    graph, Y, B, u_0, theta_star = gen_data(N, SEED)
    n = graph["n"]; BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
    t0 = time.time(); curves = {}; curves_med = {}; conv = {}
    print(f"수렴 속도 | n={N} | T_MAX={T_MAX} | seed {SEED} | AWSGLD=가중 R̂\n", flush=True)
    for m in METHODS:
        chains = []; Js = []; aw_final = None
        for ci, val in enumerate(INITS):
            ini = (np.random.RandomState(7000 + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))
            ths, J, aw = run(m, ini, Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * SEED + ci)
            chains.append(ths); Js.append(J); aw_final = aw
        wtd = (m == "AWSGLD")
        rs = [rhat_w(chains, Js, aw_final, BURN, tc, wtd) for tc in CKPTS]
        curves[m] = [x[0] for x in rs]; curves_med[m] = [x[1] for x in rs]
        below = [CKPTS[i] for i, x in enumerate(rs) if x[0] < 1.2]
        conv[m] = below[0] if below else None
        cs = "미수렴" if conv[m] is None else f"{conv[m]}"
        print(f"  {m:>8}: R̂max @ {CKPTS} = {[round(x[0],2) for x in rs]}  (med {[round(x[1],2) for x in rs]}) → 수렴 {cs}  ({int(time.time()-t0)}s)", flush=True)

    with open(os.path.join(HERE, f"awsgld_convergence_n{N}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method"] + [f"rhatmax@{c}" for c in CKPTS] + [f"rhatmed@{c}" for c in CKPTS] + ["conv_iter"])
        for m in METHODS:
            w.writerow([m] + [round(x, 3) for x in curves[m]] + [round(x, 3) for x in curves_med[m]] + [conv[m] if conv[m] else "none"])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for m in METHODS:
        ax.plot(CKPTS, curves[m], "o-", color=COL[m], lw=1.8, ms=5, label=m + (" (가중)" if m == "AWSGLD" else ""))
    ax.axhline(1.2, color="red", ls="--", lw=1.2, alpha=0.7, label="R̂=1.2")
    ax.set_xlabel("iteration (T)"); ax.set_ylabel("R̂ max (π 기준)")
    ax.set_title(f"수렴 속도 (n={N}): R̂max<1.2 도달 (seed {SEED}, 과분산 3-chain)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, f"awsgld_convergence_n{N}.png"), dpi=140, bbox_inches="tight")
    print(f"\n저장: awsgld_convergence_n{N}.png/.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
