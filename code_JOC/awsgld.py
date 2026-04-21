import numpy as np

# ===== 1. 설정 =====
np.random.seed(42)

# U(x) = -log(1/2*exp(-x^2/2)) = x^2/2 + 상수

# 표준 정규 분포 N(0, 1)
def energy_function(x):
    return 0.5 * x**2  # U(x) = x^2 / 2

def grad_energy(x):
    return x  # ∇U(x) = x

# 파라미터
n_iter = 100000   # 반복 횟수 (논문: 1e7)
tau = 1.0
zeta = 1.0
Delta_u = 0.01
m = 1000
burnin = 10000

# Adaptive weight 초기화
theta_weights = np.ones(m) / m

# 초기 상태
x_k = np.random.randn()
samples = []

print("=== AWSGLD 시작 ===")
for k in range(1, n_iter + 1):

    # learning rate & step size (논문식)
    eps_k = 0.3 / (k**0.6 + 10) # learning late
    omega_k = 0.02 / (k**0.6 + 100) # step size
    # εₖ = x (샘플) 업데이트용,
    # ωₖ = θ (가중치) 업데이트용

    # ===== Energy 및 gradient 계산 =====
    U_tilde = energy_function(x_k)
    grad_U_tilde = grad_energy(x_k) + np.random.normal(0, 0.1)  # stochastic gradient

    # 실제 SGLD/AWSGLD는 “확률적(minibatch 기반) gradient” 를 사용하는 샘플러인데,
    # 이 실험에서는 “데이터셋”이 없어서 데이터 미니배치에 의한 불확실성을 위해 gradient에 인위적으로 노이즈를 더한 것

    # ===== Subregion index =====
    J_tilde = int(np.clip(np.floor((U_tilde - (-5)) / Delta_u), 0, m - 1))

    # (U_tilde - (-5)) → 에너지 값 𝑈𝑡𝑖𝑙𝑑𝑒가 -5부터 얼마나 떨어져 있는지 계산
    # / Delta_u → 구간 크기(0.01)로 나누면 몇 번째 구간인지 소수로 나옴
    # np.floor(...) → 소수점 버리고 정수 구간 인덱스로 변환
    # np.clip(..., 0, m-1) → 만약 값이 0보다 작거나 m보다 크면 경계 안으로 “잘라냄”

    # np.clip(10, 0, 5)   # → 5
    # np.clip(-2, 0, 5)   # → 0
    # np.clip(3, 0, 5)    # → 3


    # ===== Gradient multiplier =====
    eps_log = 1e-12
    grad_multiplier = 1 + (zeta * tau / Delta_u) * (
        np.log(theta_weights[J_tilde] + eps_log)
        - np.log(theta_weights[max(J_tilde - 1, 0)] + eps_log)
    )
    grad_multiplier = np.clip(grad_multiplier, 0.5, 5.0) # 0근처나 무한대 방지

    # ===== AWSGLD 샘플링 (Eq. 8) =====
    e_k = np.random.randn() # N(0,1)
    x_k = x_k - eps_k * grad_multiplier * grad_U_tilde + np.sqrt(2 * tau * eps_k) * e_k

    # ===== Adaptive parameter 업데이트 (Eq. 9) =====
    for i in range(m):
        indicator = 1 if i >= J_tilde else 0
        theta_weights[i] += omega_k * theta_weights[J_tilde] * (indicator - theta_weights[i])

    # clip & normalize
    theta_weights = np.clip(theta_weights, 1e-10, None)
    theta_weights /= np.sum(theta_weights)

    # ===== 샘플 저장 =====
    if k > burnin:
        samples.append(x_k)

    # ===== 진행상황 출력 =====
    if k % 50000 == 0:
        print(f"Iter {k}: x={x_k:.4f}, J̃={J_tilde}, ε={eps_k:.6e}")

# ===== 3. 결과 =====
samples = np.array(samples)
print("\n=== 결과 비교 ===")
print(f"샘플 평균: {np.mean(samples):.4f}")
print(f"샘플 표준편차: {np.std(samples):.4f}")

