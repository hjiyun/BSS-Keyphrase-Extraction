"""
Study 1C — scale experiment data generator (n=100 / n=1000).

Study 1B 의 trap setup 을 그대로 가져오되 다음 두 가지를 변경
- n : 100 (소규모) 또는 1000 (대규모)
- μ_N : -1.5 (Study 1B 의 -0.8 보다 깊은 N basin → barrier ↑)
  → SGLD 계열의 N drift 약점 부각, AWSGLD 의 adaptive escape 우위 강화.

n 별로 σ_θ, p_in, p_out 을 조정 (scale 에 맞는 cluster 신호 유지).

출력
- data_n{N}_seed{S}.npz : theta_star, Y, z, A, B, u_0, conflict_mask, …

실행
- python3 simulation/study_1c/data_generator.py --n 100  [--seed 0]
- python3 simulation/study_1c/data_generator.py --n 1000 [--seed 0]
"""
import argparse
import os
import sys

import numpy as np
from scipy.linalg import solve

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_1B = os.path.join(os.path.dirname(_THIS_DIR), "study_1b")
for _p in (_THIS_DIR, STUDY_1B):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from local_trap_landscape import sigmoid  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 분포 파라미터
# ──────────────────────────────────────────────────────────────────────────
PARAMS_BASE = {
    "mu_S": 2.5, "mu_W": 1.0,
    "mu_N": -1.5,            # ← Study 1B (-0.8) 보다 깊은 N basin
    "alpha": 0.20,
    "rho_S": 0.20, "rho_W": 0.20, "rho_N": 0.60,
}

# n 에 따른 σ_θ / graph 밀도 조정
#   p_in 은 group N 내부 기대 degree 를 ~90 수준으로 유지하도록 scale 별 조정
#   (n=1500: N=900·0.10≈90, n=10000: N=6000·0.015≈90)
SCALE_CFG = {
    200:   {"sigma_theta": 0.20, "p_in": 0.50,  "p_out": 0.003},
    1500:  {"sigma_theta": 0.26, "p_in": 0.10,  "p_out": 0.001},
    10000: {"sigma_theta": 0.30, "p_in": 0.015, "p_out": 0.0002},
}

DAMPING = 0.85          # Study 1B 의 0.90 → 0.85 (prior 약화)
FLIP_RATE_S_TO_0 = 0.10  # Study 1B 의 0.30 → 0.10 (conflict 약화)
FLIP_RATE_N_TO_1 = 0.05  # Study 1B 의 0.10 → 0.05


# ──────────────────────────────────────────────────────────────────────────
# 데이터 생성 함수
# ──────────────────────────────────────────────────────────────────────────
def assign_groups(n_total, rhos, rng):
    rho_S, rho_W, rho_N = rhos
    n_S = int(round(n_total * rho_S))
    n_W = int(round(n_total * rho_W))
    n_N = n_total - n_S - n_W
    z = np.array(["S"] * n_S + ["W"] * n_W + ["N"] * n_N, dtype="<U1")
    return z[rng.permutation(n_total)], (n_S, n_W, n_N)


def sample_theta_star(z, mu_map, sigma_theta, rng):
    means = np.array([mu_map[g] for g in z], dtype=float)
    return means + rng.normal(0.0, sigma_theta, size=len(z))


def sample_Y(theta_star, alpha, rng):
    pi_star = (1.0 - alpha) * sigmoid(theta_star)
    pi_star = np.clip(pi_star, 1e-10, 1 - 1e-10)
    return rng.binomial(1, pi_star).astype(float), pi_star


def apply_label_conflict(Y, z, flip_S, flip_N, rng):
    Y_new = Y.copy()
    mask = np.zeros_like(Y, dtype=bool)
    idx_S = np.where(z == "S")[0]
    idx_N = np.where(z == "N")[0]
    n_flip_S = int(round(len(idx_S) * flip_S))
    n_flip_N = int(round(len(idx_N) * flip_N))
    if n_flip_S > 0:
        sel = rng.choice(idx_S, n_flip_S, replace=False)
        Y_new[sel] = 0.0
        mask[sel] = True
    if n_flip_N > 0:
        sel = rng.choice(idx_N, n_flip_N, replace=False)
        Y_new[sel] = 1.0
        mask[sel] = True
    return Y_new, mask


def build_sbm_graph(z, p_in, p_out, rng):
    n = len(z)
    U = rng.uniform(size=(n, n))
    same = z[:, None] == z[None, :]
    P = np.where(same, p_in, p_out)
    A = (U < P).astype(float)
    A = np.triu(A, k=1)
    A = A + A.T
    deg = A.sum(axis=1)
    for i in np.where(deg == 0)[0]:
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_B_and_u0(A, damping):
    n = A.shape[0]
    d_diag = A.sum(axis=1)
    D = np.diag(d_diag)
    G = solve(D, A)
    B = np.eye(n) - damping * G.T
    u_0 = solve(B, np.full(n, 1.0 - damping))
    return B, u_0


def generate(n, seed, out_dir):
    cfg = SCALE_CFG[n]
    params = dict(PARAMS_BASE, sigma_theta=cfg["sigma_theta"])
    rng = np.random.default_rng(seed)

    z, (n_S, n_W, n_N) = assign_groups(
        n, (params["rho_S"], params["rho_W"], params["rho_N"]), rng
    )
    mu_map = {"S": params["mu_S"], "W": params["mu_W"], "N": params["mu_N"]}
    theta_star = sample_theta_star(z, mu_map, params["sigma_theta"], rng)
    Y_clean, pi_star = sample_Y(theta_star, params["alpha"], rng)
    Y, conflict_mask = apply_label_conflict(
        Y_clean, z, FLIP_RATE_S_TO_0, FLIP_RATE_N_TO_1, rng
    )
    A = build_sbm_graph(z, cfg["p_in"], cfg["p_out"], rng)
    B, u_0 = build_B_and_u0(A, DAMPING)

    deg = A.sum(axis=1)
    npz_path = os.path.join(out_dir, f"data_n{n}_seed{seed}.npz")
    np.savez(
        npz_path,
        theta_star=theta_star, Y=Y, Y_clean=Y_clean,
        conflict_mask=conflict_mask, pi_star=pi_star,
        z=z, A=A, B=B, u_0=u_0,
        degree=deg,
        seed=np.int64(seed),
        n_total=np.int64(n),
        n_S=np.int64(n_S), n_W=np.int64(n_W), n_N=np.int64(n_N),
        damping=np.float64(DAMPING),
        p_in=np.float64(cfg["p_in"]), p_out=np.float64(cfg["p_out"]),
        flip_rate_S_to_0=np.float64(FLIP_RATE_S_TO_0),
        flip_rate_N_to_1=np.float64(FLIP_RATE_N_TO_1),
        **{f"param_{k}": v for k, v in params.items()},
    )

    print(
        f"[n={n} seed={seed}] S/W/N=({n_S},{n_W},{n_N})  "
        f"Y=1:{int(Y.sum())}/{n} (clean Y=1:{int(Y_clean.sum())})  "
        f"conflict={int(conflict_mask.sum())}  "
        f"deg mean={deg.mean():.1f}  "
        f"θ* mean=({theta_star[z=='S'].mean():+.2f},"
        f"{theta_star[z=='W'].mean():+.2f},"
        f"{theta_star[z=='N'].mean():+.2f})  -> {npz_path}"
    )
    return npz_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, choices=[200, 1500, 10000],
                        required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                        help="복수 seed 동시 생성 (예: --seeds 0 1 2 3 4)")
    args = parser.parse_args()

    cfg = SCALE_CFG[args.n]
    print(f"PARAMS: μ=({PARAMS_BASE['mu_S']},{PARAMS_BASE['mu_W']},"
          f"{PARAMS_BASE['mu_N']}) "
          f"σ_θ={cfg['sigma_theta']} α={PARAMS_BASE['alpha']} "
          f"ρ=({PARAMS_BASE['rho_S']},{PARAMS_BASE['rho_W']},"
          f"{PARAMS_BASE['rho_N']})")
    print(f"Graph: n={args.n} p_in={cfg['p_in']} p_out={cfg['p_out']} "
          f"damping={DAMPING}")
    print(f"Label conflict: flip_S_to_0={FLIP_RATE_S_TO_0} "
          f"flip_N_to_1={FLIP_RATE_N_TO_1}")
    print("-" * 76)
    for s in args.seeds:
        generate(args.n, int(s), _THIS_DIR)


if __name__ == "__main__":
    main()
