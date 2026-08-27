"""n=100 종합 지표표 — 6 샘플러 전체, 정직한 기준.

AWSGLD: π-가중(w=G[J]) 추정량으로 복원·R̂ 계산 + Kong ESS. 나머지: π 직접이라 raw.
과분산 3-chain(μ_N/μ_W/μ_S), T=10000. 데이터는 Study 1B 구성으로 n=100 새로 생성.

지표: Spearman / MSE / NDCG@10 (복원) | R̂max / R̂median (수렴) | ESS(자기상관) | KongESS(AWSGLD)
출력: full_metrics_n100.csv + 콘솔표
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
import data_generator as DG
from local_trap_landscape import PARAMS
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
N = 100
E.T = 20000; E.BURN = 2000; E.BATCH = 50
T = E.T; BURN = E.BURN; BATCH = E.BATCH; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SEED = 0; INITS = [-0.8, 1.0, 2.5, "rand"]  # 4-chain (과분산 3 + 랜덤)
METHODS = ["SGLD", "qSGLD", "cycSGLD", "SGHMC", "AWSGLD"]


def gen_data(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, ts, z


def run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); Js = np.full(T, -1); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 40)); es = []; emin = du = None; J = M_REG - 1
    for t in range(T):
        C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
        s2 = max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)
        eps = 0.3 / ((t + 1) ** 0.6 + 10)
        U = kfa.posterior_energy(Y, alpha, theta, u_0, B, s2)
        bidx = np.random.choice(n, BATCH, replace=False) if BATCH < n else None
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
        ths[t] = theta; Js[t] = J; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths, Js, aw


def ndcg_at_k(ts, th, k):
    rel = np.argsort(np.argsort(ts)).astype(float) / max(len(ts) - 1, 1)
    pred = np.argsort(th)[::-1][:k]; ideal = np.argsort(ts)[::-1][:k]
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    d = np.sum((2 ** rel[pred] - 1) * disc); i = np.sum((2 ** rel[ideal] - 1) * disc)
    return float(d / i) if i > 0 else 0.0


def rhat(means, withins, L):
    M = means.shape[0]; gm = means.mean(0)
    Bo = ((means - gm) ** 2).sum(0) / (M - 1); W = withins.mean(0)
    R = np.sqrt(np.clip(((L - 1) / L * W + Bo) / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R)), float(np.median(R))


def main():
    graph, Y, B, u_0, ts, z = gen_data(N, SEED); n = N
    BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
    L = T - BURN; rows = []; t0 = time.time()
    print(f"n={N} 종합지표 | T={T} 3-chain | AWSGLD=π가중\n", flush=True)
    for m in METHODS:
        means = []; withins = []; ess_list = []; kong = None; pooled_num = 0.0; pooled_den = 0.0
        for ci, val in enumerate(INITS):
            ini = (np.random.RandomState(7000 + 100 * SEED + ci).randn(n) * 1.5 if val == "rand" else np.full(n, float(val)))
            if m == "AWSGLD":
                ths, J, aw = run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * SEED + ci)
                post = ths[BURN:]; Jp = J[BURN:]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
                cmean = (w[:, None] * post).sum(0) / sw
                cvar = (w[:, None] * (post - cmean) ** 2).sum(0) / sw
                pooled_num += (w[:, None] * post).sum(0); pooled_den += sw
                if ci == 0: kong = (sw ** 2) / (np.sum(w ** 2) + 1e-300)
            else:
                np.random.seed(100 * SEED + ci)
                if m == "acMH":
                    ths = E.run_acmh(graph, Y, B, u_0, ini, a0, 100 * SEED + ci)
                else:
                    ths = E.run_sgld_family(m, graph, Y, B, u_0, ini, a0, 100 * SEED + ci)
                post = ths[BURN:]; cmean = post.mean(0); cvar = post.var(0, ddof=1)
                pooled_num += post.sum(0); pooled_den += post.shape[0]
            means.append(cmean); withins.append(cvar)
            ess_list.append(np.nanmedian(ess_per_node(post)))
        thetahat = pooled_num / pooled_den
        rmax, rmed = rhat(np.array(means), np.array(withins), L)
        sp = spearmanr(ts, thetahat).statistic; mse = float(np.mean((thetahat - ts) ** 2))
        KS = [5, 10, 20, 50]
        nd = {k: ndcg_at_k(ts, thetahat, k) for k in KS}
        rows.append(dict(m=m, sp=sp, mse=mse, ndcg=nd, rmax=rmax, rmed=rmed,
                         ess=float(np.median(ess_list)), kong=(float(kong) if kong else None)))
        r = rows[-1]
        print(f"  {m:>8}: Spear {sp:.3f} MSE {mse:.2f} | NDCG@5/10/20/50 {nd[5]:.3f}/{nd[10]:.3f}/{nd[20]:.3f}/{nd[50]:.3f} | "
              f"R̂max {rmax:.2f} R̂med {rmed:.3f} | ESS {r['ess']:.1f}{'' if kong is None else f' Kong {kong:.0f}'}  ({int(time.time()-t0)}s)", flush=True)

    print("\n=== n=100 NDCG@k 비교 (k 늘려도?) ===")
    print(f"{'method':>8} | {'@5':>6} {'@10':>6} {'@20':>6} {'@50':>6}")
    for r in rows:
        n = r["ndcg"]
        print(f"{r['m']:>8} | {n[5]:>6.3f} {n[10]:>6.3f} {n[20]:>6.3f} {n[50]:>6.3f}")

    print("\n=== n=100 종합 (AWSGLD=π가중, 나머지=raw) ===")
    hdr = f"{'method':>8} | {'Spear↑':>7} {'MSE↓':>6} {'NDCG50↑':>8} | {'R̂max↓':>7} {'R̂med↓':>7} | {'ESS↑':>6}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        conv = "✅" if r["rmax"] < 1.2 else "  "
        print(f"{r['m']:>8} | {r['sp']:>7.3f} {r['mse']:>6.2f} {r['ndcg'][50]:>8.3f} | {r['rmax']:>6.2f}{conv} {r['rmed']:>7.3f} | {r['ess']:>6.1f}")

    with open(os.path.join(HERE, "full_metrics_n100.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["method", "spearman", "mse", "ndcg5", "ndcg10", "ndcg20", "ndcg50", "rhat_max", "rhat_median", "ess", "kong_ess"])
        for r in rows:
            nd = r["ndcg"]
            w.writerow([r["m"], round(r["sp"], 4), round(r["mse"], 4),
                        round(nd[5], 4), round(nd[10], 4), round(nd[20], 4), round(nd[50], 4),
                        round(r["rmax"], 4), round(r["rmed"], 4), round(r["ess"], 2),
                        ("" if r["kong"] is None else round(r["kong"], 1))])
    print(f"\n저장: full_metrics_n100.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
