"""느린 노드 진단 — R̂max plateau를 붙잡는 노드는 무엇이고 왜 안 섞이나.

AWSGLD 3-chain(과분산 μ_N/μ_W/μ_S)을 n에서 돌려 노드별 가중 R̂(π 기준) 계산 →
R̂ 상위(느린) 노드의 특성(그룹 z, θ*, 라벨 Y·충돌, degree)과 어느 체인이 어긋나는지 진단.
가설: (a) 특정 그룹/극단 θ*  (b) 라벨충돌(퇴화·다봉)  (c) 그래프 hub(고degree 강결합).

인자: python3 slow_node_diag.py [N]  (기본 200)
출력: slow_node_diag_n{N}.csv + 콘솔
"""
import os, sys, time, csv
import numpy as np
from scipy.stats import invgamma
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "_archive"))
import energy_diagnostics as E
import keyphrase_functions_awsgld as kfa
import data_generator as DG
from local_trap_landscape import PARAMS

HERE = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
T = 20000; BURN = 500; BATCH = min(50, N); FLOOR = 1.0
M_REG = kfa.M_REGIONS; ZETA = kfa.ZETA; TAU = kfa.TAU; DECAY = kfa.DECAY_LR
SEED = 0; INITS = [("N", -0.8), ("W", 1.0), ("S", 2.5)]


def gen_data(n, seed):
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    theta_star = DG.sample_theta_star(z, PARAMS, rng)
    Y_clean, _ = DG.sample_Y(theta_star, PARAMS["alpha"], rng)
    Y, conflict = DG.apply_label_conflict(Y_clean, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return z, theta_star, Y.astype(float), Y_clean, conflict, A, B, u_0


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


def main():
    z, theta_star, Y, Y_clean, conflict, A, B, u_0 = gen_data(N, SEED)
    n = N; deg = A.sum(1); BtB = B.T @ B; ridge = 1e-6 * np.trace(BtB) / n
    P = np.linalg.solve(BtB + ridge * np.eye(n), np.eye(n)); P = 0.5 * (P + P.T)
    Lc = np.linalg.cholesky(P + 1e-10 * np.eye(n)); a0 = E.alpha_find(u_0, Y, E.GRID)
    t0 = time.time()
    print(f"느린 노드 진단 | n={N} T={T} seed{SEED}\n", flush=True)
    chain_wmean = []  # (M, n) 체인별 가중평균
    chain_wvar = []; aw_final = None; Jhist = []
    for ci, (nm, val) in enumerate(INITS):
        ths, J, aw = run_awsgld(np.full(n, float(val)), Y, B, u_0, a0, n, BtB, P, Lc, seed=100 * SEED + ci)
        post = ths[BURN:]; Jp = J[BURN:]; w = aw[Jp].copy(); w[Jp >= M_REG - 1] = 0.0; sw = w.sum()
        m = (w[:, None] * post).sum(0) / sw; v = (w[:, None] * (post - m) ** 2).sum(0) / sw
        chain_wmean.append(m); chain_wvar.append(v); aw_final = aw
        print(f"  chain {ci} ({nm} init) done ({int(time.time()-t0)}s)", flush=True)
    cm = np.array(chain_wmean); wv = np.array(chain_wvar); M = len(INITS); L = T - BURN
    gm = cm.mean(0); Bo = ((cm - gm) ** 2).sum(0) / (M - 1); W = wv.mean(0)
    Rn = np.sqrt(np.clip(((L - 1) / L * W + Bo) / np.maximum(W, 1e-12), 0, None))  # 노드별 R̂

    order = np.argsort(Rn)[::-1]
    print(f"\nR̂max={Rn.max():.2f}  median={np.median(Rn):.3f}  (>1.2 노드수={int((Rn>1.2).sum())}/{n})")
    print(f"\n=== 느린 노드 top 15 (R̂ 내림차순) ===")
    print(f"{'node':>4} {'R̂':>5} {'z':>2} {'θ*':>6} {'Y':>2} {'flip':>4} {'deg':>4} | 체인별 가중평균 (N/W/S init)")
    for i in order[:15]:
        fl = "*" if conflict[i] else ""
        print(f"{i:>4} {Rn[i]:>5.2f} {z[i]:>2} {theta_star[i]:>6.2f} {int(Y[i]):>2} {fl:>4} {int(deg[i]):>4} | "
              f"{cm[0][i]:>6.2f} {cm[1][i]:>6.2f} {cm[2][i]:>6.2f}")

    # 특성 요약: 느린(상위 10%) vs 나머지
    k = max(3, n // 10); slow = order[:k]; rest = order[k:]
    def grp_frac(idx, g): return float(np.mean(z[idx] == g))
    print(f"\n=== 느린 상위10%({k}개) vs 나머지 특성 ===")
    print(f"{'':>12} {'느린':>8} {'나머지':>8}")
    print(f"{'|θ*| 평균':>12} {np.mean(np.abs(theta_star[slow])):>8.2f} {np.mean(np.abs(theta_star[rest])):>8.2f}")
    print(f"{'degree 평균':>12} {np.mean(deg[slow]):>8.1f} {np.mean(deg[rest]):>8.1f}")
    print(f"{'라벨충돌율':>12} {np.mean(conflict[slow]):>8.2f} {np.mean(conflict[rest]):>8.2f}")
    print(f"{'S 비율':>12} {grp_frac(slow,'S'):>8.2f} {grp_frac(rest,'S'):>8.2f}")
    print(f"{'W 비율':>12} {grp_frac(slow,'W'):>8.2f} {grp_frac(rest,'W'):>8.2f}")
    print(f"{'N 비율':>12} {grp_frac(slow,'N'):>8.2f} {grp_frac(rest,'N'):>8.2f}")

    with open(os.path.join(HERE, f"slow_node_diag_n{N}.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["node", "rhat", "group", "theta_star", "Y", "conflict", "degree", "wmean_Ninit", "wmean_Winit", "wmean_Sinit"])
        for i in order:
            w.writerow([i, round(Rn[i], 3), z[i], round(theta_star[i], 3), int(Y[i]), int(conflict[i]), int(deg[i]),
                        round(cm[0][i], 3), round(cm[1][i], 3), round(cm[2][i], 3)])
    print(f"\n저장: slow_node_diag_n{N}.csv ({int(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
