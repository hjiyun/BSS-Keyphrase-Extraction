"""AWSGLD ζ(틸트 강도) 튜닝 — 가중(π 기준) R̂·ESS·복원 최적점 탐색.

원리: 볼록 BSS는 트랩이 없어 강한 틸트(ζ=5)가 불필요. ζ를 낮추면 ϖ가 π에 가까워져
가중 R̂·가중치 분산이 좋아진다(트랩 탈출 손실 없음). ζ 스윕으로 최적점을 찾는다.
가중치 w=G[J] (참조식). 다른 샘플러 raw(=π) R̂: qSGLD 1.37 / acMH 1.78 / cycSGLD 4.78.

출력: awsgld_zeta_tune.csv + 콘솔표
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
from extra_metrics import ess_per_node

HERE = os.path.dirname(os.path.abspath(__file__))
T = 5000; BURN = 500; BATCH = 100; FLOOR = 1.0
M_REG = kfa.M_REGIONS; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SEEDS = [0, 1, 2]
INITS = [-0.8, 1.0, 2.5]
ZETAS = [0.5, 1.0, 2.0, 3.0, 5.0]


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"], d["theta_star"]


def run_awsgld(zeta, ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); Js = np.full(T, -1); alpha = a0
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 20)); es = []; emin = du = None; J = M_REG - 1
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
            gm = float(np.clip(1 + (zeta * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; Js[t] = J; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths, Js, aw


def rhat(means, withins, L):
    M = means.shape[0]; gm = means.mean(0)
    Bo = ((means - gm) ** 2).sum(0) / (M - 1); W = withins.mean(0)
    R = np.sqrt(np.clip(((L - 1) / L * W + Bo) / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R))


def main():
    L = T - BURN; res = {z: {"wr": [], "rr": [], "ess": [], "kong": [], "wsp": [], "wmse": []} for z in ZETAS}
    t0 = time.time()
    print(f"ζ 튜닝 | seeds={SEEDS} | ζ={ZETAS} | 가중 R̂/ESS/복원\n", flush=True)
    for s in SEEDS:
        graph, Y, B, u_0, theta_star = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
        for z in ZETAS:
            rm = []; rw = []; wm = []; ww = []; ess = []; kong = []; wth0 = None
            for ci, val in enumerate(INITS):
                ths, Js, aw = run_awsgld(z, np.full(n, float(val)), Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * s + ci)
                post = ths[BURN:]; Jp = Js[BURN:]
                w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
                rm.append(post.mean(0)); rw.append(post.var(0, ddof=1)); ess.append(np.nanmedian(ess_per_node(post)))
                wmn = (w[:, None] * post).sum(0) / sw
                wm.append(wmn); ww.append((w[:, None] * (post - wmn) ** 2).sum(0) / sw)
                kong.append((sw ** 2) / (np.sum(w ** 2) + 1e-300))
                if ci == 0: wth0 = wmn
            res[z]["wr"].append(rhat(np.array(wm), np.array(ww), L))
            res[z]["rr"].append(rhat(np.array(rm), np.array(rw), L))
            res[z]["ess"].append(float(np.median(ess))); res[z]["kong"].append(float(np.median(kong)))
            res[z]["wsp"].append(spearmanr(theta_star, wth0).statistic)
            res[z]["wmse"].append(float(np.mean((wth0 - theta_star) ** 2)))
        print(f"  seed {s} done ({int(time.time()-t0)}s)", flush=True)

    print("\n=== ζ 스윕 (3-seed 평균) | 가중=π 기준 ===")
    print(f"{'ζ':>5} | {'가중R̂':>7} {'raw R̂':>7} | {'ESS':>6} {'KongESS':>8} | {'가중Spear':>9} {'가중MSE':>8}")
    print("-" * 62)
    for z in ZETAS:
        r = res[z]
        print(f"{z:>5} | {np.mean(r['wr']):>7.2f} {np.mean(r['rr']):>7.2f} | {np.mean(r['ess']):>6.1f} "
              f"{np.mean(r['kong']):>8.0f} | {np.mean(r['wsp']):>9.3f} {np.mean(r['wmse']):>8.2f}")
    print("\n비교 기준(π raw): qSGLD R̂ 1.37 / acMH 1.78 / cycSGLD 4.78 | 다른 ESS: qSGLD 13.1 / cyc 4.7")

    with open(os.path.join(HERE, "awsgld_zeta_tune.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["zeta", "weighted_rhat", "raw_rhat", "ess_autocorr", "kong_ess", "weighted_spearman", "weighted_mse"])
        for z in ZETAS:
            r = res[z]
            w.writerow([z, round(np.mean(r['wr']), 3), round(np.mean(r['rr']), 3), round(np.mean(r['ess']), 2),
                        round(np.mean(r['kong']), 1), round(np.mean(r['wsp']), 4), round(np.mean(r['wmse']), 4)])
    print(f"\n저장: awsgld_zeta_tune.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
