"""합성 S/W/N 데이터 위의 실제 n차원 log-sum-exp K-mode mixture 에너지.

study_2/trap_consensus.py 의 consensus 트랩을 합성판으로 이식한다.
- 공통 신호(진짜 키워드 = S∪W 노드)는 모든 모드 중심에서 높다 (COMMON).
- 각 모드는 고유한 미끼(N 노드의 mode별 부분집합)를 높인다 (DECOY).
→ 한 모드에 갇히면 그 모드의 미끼에 오염, 여러 모드를 평균내면 미끼는 상쇄되고
  공통 키워드만 남는다.

에너지 (샘플러가 실제 항해하는 것):
  U_k(θ)   = −loglik(Y|θ,α) + ‖B(θ − u^(k))‖² / (2σ²)
  U_mix(θ) = −logsumexp_k(−U_k(θ))

−loglik 이 모드 공통이므로:
  U_mix(θ) = −loglik(Y|θ,α)  −  logsumexp_k(−p_k),   p_k = ‖B(θ−u^(k))‖²/(2σ²)
  ∇U_mix    = ∇(−loglik)  +  BtB(θ − ū)/σ²,          ū = Σ_k softmax(−p_k) u^(k)
즉 ∇U_mix 는 kfa.grad_posterior_energy 에 u_0 자리로 softmax 가중중심 ū 를 넣은 것과 같다
(likelihood·minibatch 처리를 study_a0 와 동일 경로로 재사용).
"""
import os, sys
import numpy as np
from scipy.special import logsumexp

_HERE = os.path.dirname(os.path.abspath(__file__))
_STUDY1B = os.path.dirname(_HERE)
ROOT = os.path.dirname(os.path.dirname(_STUDY1B))
CODE = os.path.join(ROOT, "code_JOC")
for p in (_STUDY1B, CODE):
    if p not in sys.path:
        sys.path.insert(0, p)

import keyphrase_functions_awsgld as kfa            # noqa: E402
import data_generator as DG                          # study_1b/data_generator.py  # noqa: E402
from local_trap_landscape import PARAMS              # noqa: E402

# ── mixture 트랩 파라미터 (trap_consensus 최적값 이식) ──
K_MODES = 5
SIG2 = 2.0            # 고정 σ² (모드 분리 유지, CLAUDE.md: α·σ² 고정 → U 는 θ만의 함수)
COMMON = 2.5          # 진짜 키워드(S∪W) 를 모든 모드에서 끌어올리는 값
DECOY = 4.0           # 각 모드의 미끼(N 부분집합) 값
BASE = -0.5           # 그 외 노드의 중심 기준선
ALPHA = PARAMS["alpha"]   # 0.20 고정


def inv_logit(x):
    return kfa.inv_logit(x)


def gen(n, seed):
    """study_1b data_generator 와 동일 경로 (θ*, Y, B, u_0, z)."""
    rng = np.random.default_rng(seed)
    z, _ = DG.assign_groups(n, (PARAMS["rho_S"], PARAMS["rho_W"], PARAMS["rho_N"]), rng)
    ts = DG.sample_theta_star(z, PARAMS, rng)
    Yc, _ = DG.sample_Y(ts, PARAMS["alpha"], rng)
    Y, _ = DG.apply_label_conflict(Yc, z, DG.FLIP_RATE_S_TO_0, DG.FLIP_RATE_N_TO_1, rng)
    A = DG.build_sbm_graph(z, DG.P_IN, DG.P_OUT, rng)
    B, u_0, _ = DG.build_B_and_u0(A, DG.DAMPING)
    return {"n": n, "A": A, "D": np.diag(A.sum(1))}, Y.astype(float), B, u_0, ts, z


def build_centers(z, seed=0):
    """K개 모드 중심 u^(k). 공통 신호=S∪W, 미끼=N 노드를 K등분한 mode별 부분집합."""
    n = len(z)
    kw = np.where((z == "S") | (z == "W"))[0]          # 진짜 키워드(공통 신호)
    nonkw = np.where(z == "N")[0]
    rng = np.random.RandomState(seed)
    perm = rng.permutation(nonkw)
    decoys = [perm[i::K_MODES] for i in range(K_MODES)]  # 모드별 미끼
    centers = np.full((K_MODES, n), float(BASE))
    for k in range(K_MODES):
        centers[k, kw] = COMMON
        centers[k, decoys[k]] = DECOY
    return centers, kw, nonkw, decoys


def loglik_energy(Y, theta, alpha=ALPHA):
    """−loglik(Y|θ,α) (mixture 공통항)."""
    temp = np.clip((1 - alpha) * np.clip(inv_logit(theta), 1e-10, 1 - 1e-10), 1e-10, 1 - 1e-10)
    return -np.sum(Y * np.log(temp) + (1 - Y) * np.log(1 - temp))


class MixtureEnergy:
    """U_mix, ∇U_mix, 현재 모드 argmin_k p_k 를 제공."""

    def __init__(self, Y, B, centers, sig2=SIG2, alpha=ALPHA):
        self.Y = Y; self.B = B; self.centers = centers
        self.sig2 = sig2; self.alpha = alpha
        self.BtB = B.T @ B
        self.n = B.shape[0]

    def _pk(self, theta):
        d = theta[None, :] - self.centers          # K×n
        Bd = d @ self.B.T                          # K×n
        return 0.5 * np.sum(Bd * Bd, axis=1) / self.sig2   # (K,)

    def energy(self, theta):
        pk = self._pk(theta)
        return loglik_energy(self.Y, theta, self.alpha) - logsumexp(-pk)

    def energy_trace(self, theta_store):
        """(T,n) → (T,) U_mix 벡터화."""
        d = theta_store[:, None, :] - self.centers[None, :, :]      # T×K×n
        Bd = d @ self.B.T                                          # T×K×n
        pk = 0.5 * np.sum(Bd * Bd, axis=2) / self.sig2             # T×K
        prior = -logsumexp(-pk, axis=1)                           # T
        temp = np.clip((1 - self.alpha) * np.clip(inv_logit(theta_store), 1e-10, 1 - 1e-10),
                       1e-10, 1 - 1e-10)
        loglik = np.sum(self.Y[None, :] * np.log(temp) + (1 - self.Y)[None, :] * np.log(1 - temp), axis=1)
        return -loglik + prior

    def ubar(self, theta):
        pk = self._pk(theta)
        w = np.exp(-pk - logsumexp(-pk))          # softmax(−p_k)
        return w @ self.centers                    # (n,)

    def grad(self, theta, batch_idx=None):
        """∇U_mix = kfa.grad_posterior_energy(u_0=ū)."""
        ub = self.ubar(theta)
        return kfa.grad_posterior_energy(self.Y, self.alpha, theta, ub, self.B, self.sig2,
                                         batch_idx=batch_idx, BtB=self.BtB)

    def mode(self, theta):
        return int(np.argmin(self._pk(theta)))
