"""AWSGLD 중요도 가중 진단 — 원표본(ϖ) vs 가중(π) 지표 비교.

심사자 지적: AWSGLD 표본은 틸트된 ϖ ∝ π/Ψ^ζ 에서 나오므로 raw R̂/ESS/θ̂ 는 ϖ 기준.
π 기준 주장을 하려면 중요도 가중이 필요하다. 참조 구현(Adaptively-Weighted-...-main,
sgmcmc.py:172)의 importance weight = G[J] (수렴 adaptive_weights[밴드], 최상위 밴드는 0)을
그대로 사용해, 정규화 가중평균으로 π 추정량을 만든다. (다른 샘플러는 π 직접 표본이라 가중 불필요.)

산출: AWSGLD 에 대해 raw vs 가중
  - 복원: Spearman / MSE (vs θ*)
  - 수렴: R̂max (raw θ표본 vs 가중 추정량)
  - 효율: raw ESS(자기상관) vs Kong ESS(가중분산) = (Σw)²/Σw²
비교용으로 다른 샘플러의 raw 값은 기존 파일(strip6_cutoff.csv, convergence_cutoff.csv) 인용.
출력: awsgld_weighted_diag.csv + 콘솔표
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
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SEEDS = [0, 1, 2]
INITS = [-0.8, 1.0, 2.5]   # 과분산 3 시작점 (μ_N/μ_W/μ_S)


def load(seed):
    d = np.load(os.path.join(HERE, f"data_seed{seed}.npz"))
    n = int(d["n_total"]); A = d["A"]; graph = {"n": n, "A": A, "D": np.diag(A.sum(1))}
    return graph, d["Y"].astype(float), d["B"], d["u_0"], d["theta_star"]


def run_awsgld(ini, Y, B, u_0, a0, n, BtB, P, Lc, seed):
    """AWSGLD run + 밴드 J 기록 + 최종 adaptive_weights 반환."""
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
            gm = float(np.clip(1 + (ZETA * TAU / du) * (np.log(aw[J] + 1e-12) - np.log(aw[J - 1] + 1e-12)), 0.1, 10.0))
        theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * TAU * eps) * (Lc @ np.random.randn(n))
        theta = np.clip(theta, -700, 700)
        if t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; Js[t] = J; alpha = E.alpha_find(theta, Y, E.GRID)
    return ths, Js, aw


def weights_for(Js_post, aw_final):
    """참조식 importance weight w = G[J] (최상위 밴드 J=M-1 은 0)."""
    w = aw_final[Js_post].copy()
    w[Js_post >= M_REG - 1] = 0.0
    return w


def rhat_from_chainstats(means, withins, L):
    """chain별 (mean, within-var) 로 노드별 R̂max. means/withins: (M, n)."""
    M = means.shape[0]; gm = means.mean(0)
    B_over_L = ((means - gm) ** 2).sum(0) / (M - 1)
    W = withins.mean(0)
    R = np.sqrt(np.clip(((L - 1) / L * W + B_over_L) / np.maximum(W, 1e-12), 0, None))
    return float(np.nanmax(R))


def main():
    rows = []
    t0 = time.time()
    print(f"AWSGLD raw vs 가중 진단 | seeds={SEEDS} | 3 chains | w=G[J] (참조식)\n", flush=True)
    for s in SEEDS:
        graph, Y, B, u_0, theta_star = load(s); n = graph["n"]
        BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
        P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
        Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
        raw_means = []; raw_within = []; w_means = []; w_within = []
        raw_ess = []; kong = []; raw_thetahat = []; w_thetahat = []
        for ci, val in enumerate(INITS):
            ths, Js, aw = run_awsgld(np.full(n, float(val)), Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * s + ci)
            post = ths[BURN:]; Jp = Js[BURN:]
            w = weights_for(Jp, aw); sw = w.sum()
            # raw
            raw_means.append(post.mean(0)); raw_within.append(post.var(0, ddof=1))
            raw_ess.append(np.nanmedian(ess_per_node(post)))
            raw_thetahat.append(post.mean(0))
            # weighted (per-node 가중평균/가중분산; w 는 시간축 공통)
            wm = (w[:, None] * post).sum(0) / sw
            wv = (w[:, None] * (post - wm) ** 2).sum(0) / sw
            w_means.append(wm); w_within.append(wv); w_thetahat.append(wm)
            kong.append((sw ** 2) / (np.sum(w ** 2) + 1e-300))   # Kong ESS (시간축 공통 → 스칼라)
        raw_means = np.array(raw_means); raw_within = np.array(raw_within)
        w_means = np.array(w_means); w_within = np.array(w_within)
        L = T - BURN
        # 복원: chain0(=bad init μ_N) 추정량 vs θ* (기존 지표와 동일 관행)
        rth = raw_thetahat[0]; wth = w_thetahat[0]
        raw_sp = spearmanr(theta_star, rth).statistic; raw_mse = float(np.mean((rth - theta_star) ** 2))
        w_sp = spearmanr(theta_star, wth).statistic; w_mse = float(np.mean((wth - theta_star) ** 2))
        raw_rhat = rhat_from_chainstats(raw_means, raw_within, L)
        w_rhat = rhat_from_chainstats(w_means, w_within, L)
        rows.append(dict(seed=s, raw_sp=raw_sp, w_sp=w_sp, raw_mse=raw_mse, w_mse=w_mse,
                         raw_rhat=raw_rhat, w_rhat=w_rhat,
                         raw_ess=float(np.median(raw_ess)), kong=float(np.median(kong)),
                         wmin=float(np.min([weights_for(Js[BURN:], aw).min() for _ in [0]])),
                         wmax=float(aw.max())))
        r = rows[-1]
        print(f"  seed {s}: Spearman raw {raw_sp:.3f}→가중 {w_sp:.3f} | MSE raw {raw_mse:.2f}→가중 {w_mse:.2f} "
              f"| R̂ raw {raw_rhat:.2f}→가중 {w_rhat:.2f} | ESS raw {r['raw_ess']:.1f} KongESS {r['kong']:.0f} "
              f"({int(time.time()-t0)}s)", flush=True)

    # 평균
    def mean(k): return float(np.mean([r[k] for r in rows]))
    print("\n=== AWSGLD 평균 (raw → 가중, 3 seed) ===")
    print(f"  Spearman : {mean('raw_sp'):.3f}  →  {mean('w_sp'):.3f}")
    print(f"  MSE      : {mean('raw_mse'):.2f}  →  {mean('w_mse'):.2f}")
    print(f"  R̂max     : {mean('raw_rhat'):.2f}  →  {mean('w_rhat'):.2f}")
    print(f"  ESS(자기상관) {mean('raw_ess'):.1f}  |  Kong ESS(가중분산) {mean('kong'):.0f}  (raw 표본수 {T-BURN})")
    print("\n  [비교용] 다른 샘플러 raw ESS (strip6_cutoff.csv): qSGLD 13.1 / acMH 7.6 / SGHMC 7.3 / SGLD 5.6 / cycSGLD 4.7")

    with open(os.path.join(HERE, "awsgld_weighted_diag.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(list(rows[0].keys()))
        for r in rows: w.writerow([round(v, 4) if isinstance(v, float) else v for v in r.values()])
    print(f"\n저장: awsgld_weighted_diag.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
