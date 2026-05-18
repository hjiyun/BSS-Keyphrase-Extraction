"""
Study 1B — Target 분포에서 실제 사용할 데이터 생성.

설계
----
입력  : local_trap_landscape.py 의 PARAMS (mu_S, mu_W, mu_N, sigma_theta,
        alpha, rho_S/W/N).
출력  : seed 별 npz 파일에 theta_star, Y, z, A, B, u_0 저장.

핵심 동작
---------
(a) Group 할당 (z)
    n=400 phrase 중 ρ=(0.2, 0.2, 0.6) 비율로 80/80/240 → S/W/N.
    permutation 후 z 에 저장. 이는 truth (sampler 에게는 숨김).

(b) θ* 생성 (target 분포 sampling)
    z_i = g 인 phrase 에 대해 θ_i^* ~ N(mu_g, sigma_theta^2).
    이게 sampler 가 맞춰야 할 정답.

(c) 관측 Y 생성 (BSS 모형)
    Y_i ~ Bernoulli((1-alpha) sigmoid(θ_i^*)).
    이게 sampler 가 보는 유일한 라벨 신호.

(d) Graph A 생성 (SBM)
    같은 group: p_in=0.30, 다른 group: p_out=0.05.
    대칭화 후 isolated node 가 있으면 임의 노드와 연결.
    BSS 의 graph prior 가 받는 입력.

(e) BSS 입력 계산
    D = diag(degree(A))
    B   = I - d · (D^-1 A)^T            (langevin_methods_comparison.build_B)
    u_0 = solve(B, (1-d) · 1_n)         (BSS 의 prior mean)
    여기서 d = DAMPING = 0.85.

CLAUDE.md 규칙
- 시드 고정 (재현성).
- Truth 인 theta_star 는 sampler 가 보지 않는다는 의미로 npz 안에는 함께
  저장하지만, 평가 단계에서만 unpack 한다.
- u_0 = solve(B) 사용. Oracle 초기화 금지 → u_0 가 theta_star 와 무관한 형태.

실행
----
python3 simulation/study_1b/data_generator.py            # 기본 시드 0..4
python3 simulation/study_1b/data_generator.py 7 11 13    # 임의 시드
"""
import os
import sys

import numpy as np
from scipy.linalg import solve

# local_trap_landscape.PARAMS 를 단일 진실 원천으로 사용한다.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)
from local_trap_landscape import PARAMS, sigmoid  # noqa: E402


# ──────────────────────────────────────────────────────────────────────────
# 데이터 생성 파라미터 (graph 관련)
# ──────────────────────────────────────────────────────────────────────────
N_TOTAL = 400
P_IN = 0.40      # same group edge prob (강결합)
P_OUT = 0.005    # cross group edge prob (cluster 간 분리)
DAMPING = 0.90   # BSS B = I - d (D^-1 A)^T 의 d  (graph prior 영향력 ↑)
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

# Label conflict (PU-noise 위에 추가) — real BSS trap 유도용.
# S/W phrase 의 일부를 강제로 Y=0 으로, N phrase 의 일부를 강제로 Y=1 로 flip.
# graph 가 cluster 를 묶는데 Y 신호가 cluster 와 충돌 → joint posterior 가 multimodal.
FLIP_RATE_S_TO_0 = 0.30
FLIP_RATE_N_TO_1 = 0.10


def assign_groups(n_total, rhos, rng):
    """ρ 비율대로 정확한 개수의 S/W/N 라벨을 만들고 permutation."""
    rho_S, rho_W, rho_N = rhos
    n_S = int(round(n_total * rho_S))
    n_W = int(round(n_total * rho_W))
    n_N = n_total - n_S - n_W
    z = np.array(["S"] * n_S + ["W"] * n_W + ["N"] * n_N, dtype="<U1")
    z = z[rng.permutation(n_total)]
    return z, (n_S, n_W, n_N)


def sample_theta_star(z, params, rng):
    """θ_i^* ~ N(mu_{z_i}, sigma_theta^2)."""
    mu_map = {"S": params["mu_S"], "W": params["mu_W"], "N": params["mu_N"]}
    means = np.array([mu_map[g] for g in z], dtype=float)
    return means + rng.normal(0.0, params["sigma_theta"], size=len(z))


def sample_Y(theta_star, alpha, rng):
    """Y_i ~ Bernoulli((1-alpha) sigmoid(theta_i^*))."""
    pi_star = (1.0 - alpha) * sigmoid(theta_star)
    pi_star = np.clip(pi_star, 1e-10, 1 - 1e-10)
    return rng.binomial(1, pi_star).astype(float), pi_star


def apply_label_conflict(Y, z, flip_S_to_0, flip_N_to_1, rng):
    """
    Y 위에 label conflict 추가.
    - S phrase 의 flip_S_to_0 비율 → Y=0 으로 강제 (관측 누락)
    - N phrase 의 flip_N_to_1 비율 → Y=1 으로 강제 (false positive)
    반환: (Y_new, conflict_mask)  conflict_mask=True 인 노드가 flip 된 노드.
    """
    Y_new = Y.copy()
    conflict_mask = np.zeros_like(Y, dtype=bool)
    idx_S = np.where(z == "S")[0]
    n_flip_S = int(round(len(idx_S) * flip_S_to_0))
    if n_flip_S > 0:
        sel = rng.choice(idx_S, size=n_flip_S, replace=False)
        Y_new[sel] = 0.0
        conflict_mask[sel] = True
    idx_N = np.where(z == "N")[0]
    n_flip_N = int(round(len(idx_N) * flip_N_to_1))
    if n_flip_N > 0:
        sel = rng.choice(idx_N, size=n_flip_N, replace=False)
        Y_new[sel] = 1.0
        conflict_mask[sel] = True
    return Y_new, conflict_mask


def build_sbm_graph(z, p_in, p_out, rng):
    """SBM: same-group p_in, cross-group p_out. 대칭 + isolated node 보정."""
    n = len(z)
    # 상삼각만 샘플링 후 대칭화 → O(n^2) 이지만 n=200 이면 충분히 빠름.
    U = rng.uniform(size=(n, n))
    same = z[:, None] == z[None, :]
    P = np.where(same, p_in, p_out)
    A = (U < P).astype(float)
    A = np.triu(A, k=1)
    A = A + A.T

    deg = A.sum(axis=1)
    iso = np.where(deg == 0)[0]
    for i in iso:
        j = int(rng.integers(0, n))
        while j == i:
            j = int(rng.integers(0, n))
        A[i, j] = 1.0
        A[j, i] = 1.0
    return A


def build_B_and_u0(A, damping):
    """B = I - d (D^-1 A)^T,  u_0 = solve(B, (1-d) 1_n)."""
    n = A.shape[0]
    d_diag = A.sum(axis=1)
    D = np.diag(d_diag)
    G = solve(D, A)
    B = np.eye(n) - damping * G.T
    u_0 = solve(B, np.full(n, 1.0 - damping))
    return B, u_0, D


def generate_one(seed, out_dir, params):
    rng = np.random.default_rng(seed)

    z, (n_S, n_W, n_N) = assign_groups(
        N_TOTAL, (params["rho_S"], params["rho_W"], params["rho_N"]), rng
    )
    theta_star = sample_theta_star(z, params, rng)
    Y_clean, pi_star = sample_Y(theta_star, params["alpha"], rng)
    Y, conflict_mask = apply_label_conflict(
        Y_clean, z, FLIP_RATE_S_TO_0, FLIP_RATE_N_TO_1, rng
    )
    A = build_sbm_graph(z, P_IN, P_OUT, rng)
    B, u_0, D = build_B_and_u0(A, DAMPING)

    deg = A.sum(axis=1)
    npz_path = os.path.join(out_dir, f"data_seed{seed}.npz")
    np.savez(
        npz_path,
        theta_star=theta_star,
        Y=Y,
        Y_clean=Y_clean,
        conflict_mask=conflict_mask,
        pi_star=pi_star,
        z=z,
        A=A,
        B=B,
        u_0=u_0,
        degree=deg,
        seed=np.int64(seed),
        n_total=np.int64(N_TOTAL),
        n_S=np.int64(n_S), n_W=np.int64(n_W), n_N=np.int64(n_N),
        damping=np.float64(DAMPING),
        p_in=np.float64(P_IN), p_out=np.float64(P_OUT),
        flip_rate_S_to_0=np.float64(FLIP_RATE_S_TO_0),
        flip_rate_N_to_1=np.float64(FLIP_RATE_N_TO_1),
        **{f"param_{k}": v for k, v in params.items()},
    )

    n_pos = int(Y.sum())
    n_conflict = int(conflict_mask.sum())
    print(
        f"[seed {seed:3d}] n=({n_S},{n_W},{n_N})  "
        f"Y=1:{n_pos}/{N_TOTAL} (clean Y=1:{int(Y_clean.sum())})  "
        f"conflict={n_conflict}  "
        f"degree mean={deg.mean():.1f} (min={int(deg.min())}, max={int(deg.max())})  "
        f"theta* mean=({theta_star[z=='S'].mean():+.2f},"
        f"{theta_star[z=='W'].mean():+.2f},"
        f"{theta_star[z=='N'].mean():+.2f})  "
        f"-> {npz_path}"
    )
    return npz_path


def main(seeds):
    out_dir = _THIS_DIR
    print(
        f"PARAMS: mu=({PARAMS['mu_S']},{PARAMS['mu_W']},{PARAMS['mu_N']})  "
        f"sigma_theta={PARAMS['sigma_theta']}  alpha={PARAMS['alpha']}  "
        f"rho=({PARAMS['rho_S']},{PARAMS['rho_W']},{PARAMS['rho_N']})"
    )
    print(f"Graph: n={N_TOTAL}  p_in={P_IN}  p_out={P_OUT}  damping={DAMPING}")
    print(f"Label conflict: flip_S_to_0={FLIP_RATE_S_TO_0}  "
          f"flip_N_to_1={FLIP_RATE_N_TO_1}")
    print("-" * 76)
    for s in seeds:
        generate_one(int(s), out_dir, PARAMS)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        seeds = [int(x) for x in sys.argv[1:]]
    else:
        seeds = DEFAULT_SEEDS
    main(seeds)
