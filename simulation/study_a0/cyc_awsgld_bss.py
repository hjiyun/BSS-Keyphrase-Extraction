"""cyc-AWSGLD 를 BSS 원 에너지 함수(Study 1B 실데이터)에서 테스트.

BSS U(θ) = −loglik(Y|θ,α) + ‖B(θ−u_0)‖²/(2σ²)  (near-convex, 얕은 트랩)
목표는 '낮은 에너지'가 아니라 정답 θ* 복원(Spearman/MSE) + 시작점 강건성.

비교: AWSGLD / cycSGLD / cyc-AWSGLD (플랫히스토그램 + 순환 스케줄).
여러 시작점(μ_N, μ_W, μ_S, 랜덤×2)에서 실행 → 시작점 의존성 확인.
지표: Spearman(θ̂,θ*), MSE_all, 도달 min U.
"""
import os, sys, time
import numpy as np
from scipy.stats import spearmanr, invgamma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa

d = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_seed0.npz"))
theta_star = d["theta_star"]
graph, Y, B, u_0, z = E.load(); a = E.alpha_find(u_0, Y, E.GRID); n = graph["n"]
BtB = B.T @ B
ridge = 1e-6 * np.trace(BtB) / n
P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n))
T = 5000; BURN = 500; BATCH = 100; FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
CYC = 10


def _sigma2(theta):
    C = (B @ (theta - u_0)) @ (B @ (theta - u_0))
    return max(invgamma.rvs(n / 2 + 0.001, scale=C / 2 + 0.001), FLOOR)


def run(method, ini, seed):
    """method ∈ {AWSGLD, cycSGLD, cycAWSGLD}. flat-histogram = AWSGLD/cycAWSGLD."""
    np.random.seed(seed); theta = ini.copy(); ths = np.zeros((T, n)); alpha = a
    aw = np.arange(1, M_REG + 1, dtype=float) / M_REG
    warm = min(100, max(10, T // 20)); es = []; emin = du = None; J = M_REG - 1
    use_aw = method in ("AWSGLD", "cycAWSGLD")
    for t in range(T):
        s2 = _sigma2(theta)
        # 순환 스케줄 (cyc 계열)
        if method in ("cycSGLD", "cycAWSGLD"):
            cl = max(1, T // CYC); beta = (t % cl) / cl
            eps = (0.3 if use_aw else 0.02) / 2.0 * (np.cos(np.pi * min(beta, 0.8)) + 1)
            tau_k = TAU if beta >= 0.8 else TAU / 1e4
        else:  # AWSGLD
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
        if method in ("AWSGLD", "cycAWSGLD"):
            theta = theta - eps * gm * (P @ gU) + np.sqrt(2 * tau_k * eps) * (Lc @ np.random.randn(n))
        else:  # cycSGLD: 무전처리
            theta = theta - eps * gU + np.sqrt(2 * tau_k * eps) * np.random.randn(n)
        theta = np.clip(theta, -700, 700)
        if use_aw and t >= warm:
            dec = min(1.0, DECAY / ((t + 1) ** 0.75 + 1000)); cw = aw[J]
            aw[J:] = aw[J:] + dec * cw * (1 - aw[J:]); aw[:J] = aw[:J] - dec * cw * aw[:J]; aw = np.clip(aw, 1e-10, 1)
        ths[t] = theta; alpha = E.alpha_find(theta, Y, E.GRID)
    th_hat = ths[BURN:].mean(0)
    U_trace = E.energy_trace_common(ths, Y, B, u_0, a)
    return dict(spearman=spearmanr(theta_star, th_hat).statistic,
                mse=float(np.mean((th_hat - theta_star) ** 2)),
                minU=float(U_trace[BURN:].min()))


def main():
    inits = {"μ_N (bad)": np.full(n, -0.8), "μ_W": np.full(n, 1.0), "μ_S": np.full(n, 2.5)}
    rng = np.random.RandomState(7)
    inits["random#1"] = rng.randn(n); inits["random#2"] = rng.randn(n)
    METHODS = ["AWSGLD", "cycSGLD", "cycAWSGLD"]
    print(f"BSS 원 에너지 U(θ) 에서 cyc-AWSGLD 테스트 | n={n}, T={T}\n", flush=True)
    print(f"{'init':>12} | " + " | ".join(f"{m:>26}" for m in METHODS))
    print(f"{'':>12} | " + " | ".join(f"{'Spear  MSE   minU':>26}" for m in METHODS))
    print("-" * 100)
    agg = {m: {"sp": [], "mse": [], "minU": []} for m in METHODS}
    t0 = time.time()
    for name, ini in inits.items():
        cells = []
        for m in METHODS:
            r = run(m, ini, seed=0)
            agg[m]["sp"].append(r["spearman"]); agg[m]["mse"].append(r["mse"]); agg[m]["minU"].append(r["minU"])
            cells.append(f"{r['spearman']:>6.3f} {r['mse']:>5.2f} {r['minU']:>6.0f}")
        print(f"{name:>12} | " + " | ".join(f"{c:>26}" for c in cells), flush=True)
    print("-" * 100)
    print(f"{'평균':>12} | " + " | ".join(
        f"{np.mean(agg[m]['sp']):>6.3f} {np.mean(agg[m]['mse']):>5.2f} {np.mean(agg[m]['minU']):>6.0f}".rjust(26)
        for m in METHODS))
    print(f"{'Spear 표준편차':>12} | " + " | ".join(f"{np.std(agg[m]['sp']):>26.3f}" for m in METHODS))
    print(f"\n({int(time.time()-t0)}s) — Spear 표준편차 작을수록 시작점 강건. minU 낮을수록 깊이 하강(≠좋음).")


if __name__ == "__main__":
    main()
