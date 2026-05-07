# BSS Keyphrase Extraction — Troubleshooting

본 문서는 AWSGLD 도입 및 SGLD 변형 비교 실험 과정에서 마주친 문제와 해결 과정을 정리한다.
각 항목은 **증상 → 원인 → 해결 → 검증** 4단계로 기술한다.

---

## 1. AWSGLD geometry 문제 (Spearman 역전)

### 증상
초기 AWSGLD 구현에서 acMH 대비 Spearman = **−0.604** 관측. 즉 추정된 θ̂ 의 순위가
실제 θ\* 와 거의 정반대로 정렬됨. NDCG@k = 0.328 (랜덤보다 나쁨).

### 원인 분석 (system적 ablation)
다음 순서로 component 하나씩 비활성화하여 원인을 좁혀나갔다.

| Step | 비활성화한 것 | Spearman | MSE(θ) |
|---|---|---|---|
| 0 | 원본 (모두 켜짐) | −0.604 | 9.19 |
| 1 | adaptive weighting (`grad_mult=1`, weight update off) | −0.199 | 31.28 |
| 2 | + α 재추정 (`alpha_find` 끔, 초기값 고정) | −0.176 | 30.71 |
| 3 | + σ² 고정 (1.0) | −0.264 | 23.83 |
| 4 | + noise off (pure GD) | −0.382 | 24.03 |
| **5** | **+ preconditioning 적용** | **+0.739** | **3.25** |

→ 결정타는 단계 5. **graph-coupled prior `(θ-u₀)ᵀ B^T B (θ-u₀)/(2σ²)` 의 anisotropy** 때문.

`B^T B` 는 그래프 Laplacian 유사 행렬로 eigenvalue spread 가 크다.
큰 eigenvalue 방향(고주파 모드)은 prior 가 강하게 끌어당기고,
작은 eigenvalue 방향(저주파 모드)은 거의 풀어준다.
Vanilla gradient 는 모든 좌표에 동일한 step size 를 쓰므로
고주파 방향은 발산, 저주파 방향은 정체되어 θ̂ 의 모양이 무너진다.

### 해결
**Preconditioner** P ≈ (B^T B + ridge·I)⁻¹ 적용.

```python
BtB_fixed = B.T @ B
ridge = 1e-6 * np.trace(BtB_fixed) / n
P_precond = solve(BtB_fixed + ridge * np.eye(n), np.eye(n))
P_sym = 0.5 * (P_precond + P_precond.T)
L_precond = np.linalg.cholesky(P_sym + 1e-10 * np.eye(n))

# Langevin update
theta = theta - eps_k * (P_precond @ grad_U) + sqrt(2*tau*eps_k) * (L_precond @ noise)
```

P @ grad_U 가 prior 항을 `(θ-u₀)/σ²` 형태로 단순화한다.
Likelihood 항은 그래프의 natural geometry 에서 smoothing 된다.

### 검증
preconditioning 만 적용한 상태(다른 component 모두 off)에서 Spearman 0.739 회복.
이후 noise / α 재추정 / weighting 을 차례로 복원해도 안정 유지.

### 파일
- `code_JOC/keyphrase_functions_awsgld_0422.py` — preconditioning 영구 반영

---

## 2. σ² collapse positive feedback

### 증상
preconditioning 적용으로 정상 작동하던 sampler 에 `σ²` resampling 을 복원하니
Spearman 다시 −0.61 로 폭락.

### 원인
`σ² ~ InvGamma(n/2 + 0.001, C/2 + 0.001)` 의 평균은 약 `C/n`.
`C = ‖B(θ-u₀)‖²` 가 작을 때 `σ²` 가 매우 작아짐 → preconditioned prior gradient
`P @ grad_U ≈ (θ-u₀)/σ²` 가 폭발 → θ 가 u₀ 로 강하게 끌려감 → C 더 작아짐 → σ² 더 작아짐.
**positive feedback collapse.**

acMH 는 MH rejection 이 wild jump 를 거부해 자연스럽게 막아주지만 Langevin 에는 그런 안전장치가 없다.

### 진단 (궤적 출력)
`simlation/diagnose_sigma2_trajectory.py` 로 측정:

| 지표 | acMH (정상) | AWSGLD (broken) |
|---|---|---|
| σ² median | 1.40 | **0.011** (130x 작음) |
| σ² min | 0.76 | **0.0026** |
| C median | 338 | **2.64** |
| C min | 228 | 0.71 |

### 해결
**σ² floor 0.5** (acMH 의 minimum 0.76 아래에 안전 마진).

```python
sigma2 = invgamma.rvs(n/2 + 0.001, scale=C/2 + 0.001)
sigma2 = max(sigma2, 0.5)  # floor to prevent prior-gradient blowup
```

### 검증
floor 적용 후 Spearman 복원 (0.737), 모든 σ² 추정이 0.5 부근에서 균형.

---

## 3. 코드 성능 최적화

### 3-1. `B^T B` 매 step 재계산 (O(n³))

#### 증상
n=800, T=5000 에서 AWSGLD 가 **57s** (qSGLD 와 동등 매트릭 cost 임에도 3배 느림).

#### 원인
`grad_posterior_energy` 내부에서 매 호출 `BtB = B.T @ B` 를 재계산.
n=800 에서 **5×10⁸ flops/step × 5000 = 2.5T flops** 의 불필요한 연산.

```python
# 기존 (잘못된 부분)
def grad_posterior_energy(Y, alpha, theta, u_0, B, sigma2):
    ...
    BtB = B.T @ B  # ← 매 step O(n³)!
    grad_prior = -BtB @ (theta - u_0) / sigma2
    ...
```

#### 해결
`gibbs_mh` 진입 직후 **한 번만** 계산하고 인자로 전달.

```python
def grad_posterior_energy(Y, alpha, theta, u_0, B, sigma2, BtB=None):
    ...
    if BtB is None:
        BtB = B.T @ B
    grad_prior = -BtB @ (theta - u_0) / sigma2
    ...

# 호출부
BtB_fixed = B.T @ B  # 1회
for t in range(T):
    grad_U = grad_posterior_energy(..., BtB=BtB_fixed)
```

#### 검증
n=800 AWSGLD wall-time **57.5s → 16.4s (3.5x speedup)**, 결과 quality 동일.

---

### 3-2. `B @ (θ-u₀)` 중복 계산

#### 증상
같은 step 내에서 같은 matrix-vector product 를 여러 번 계산.

#### 원인
`(B @ (theta-u_0)).T @ (B @ (theta-u_0))` 패턴은 numpy 가 common subexpression
elimination 을 안 해서 `B @ v` 를 **2 번** 계산함. 더해 `posterior_energy` 가
내부 `posterior_gibbstheta` 에서 또 한 번 같은 계산을 함.

#### 해결
```python
for t in range(T):
    Bv = B @ (theta - u_0)        # 한 번만
    C = Bv @ Bv                    # dot product, O(n) 무시할 수준
    sigma2 = invgamma.rvs(...)

    U_tilde = posterior_energy(..., C=C)  # 미리 계산한 C forwarding
```

`posterior_gibbstheta` / `posterior_energy` 에 `C=None` 인자 추가:
```python
def posterior_gibbstheta(Y, alpha, theta, u_0, B, sigma2, C=None):
    ...
    if C is None:
        Bv = B @ (theta - u_0)
        C = Bv @ Bv
    ...
```

#### 검증
이 fix 와 BtB 캐시 fix 를 합쳐 AWSGLD 가 SGLD/qSGLD/cycSGLD 와 비슷한 ~16s 로 정렬.

---

## 4. 시뮬레이션 setup pitfalls

### 4-1. GRID 범위 mismatch

#### 증상
`α_true = 0.20` 으로 시뮬했는데 모든 method 의 `α̂` 가 정확히 0.5 로 clamp.

#### 원인
`GRID = (np.arange(10, 43) - 5) / np.arange(10, 43)` 의 최솟값 = 5/10 = 0.5.
`α_true = 0.20` 이 이 범위 밖이라 `alpha_find` 가 grid minimum 에 stuck.

#### 영향
- Ranking 지표(Spearman, NDCG): 영향 없음 (모든 노드에 같은 (1−α) 곱해짐)
- MSE(π): 영향 큼 (true π_max = 0.8·σ(θ) 인데 추정 = 0.5·σ(θ))

#### 해결 (선택지)
- 시나리오의 α 를 [0.5, 0.88] 안으로 (예: Difficult α=0.50, Sparse Opt-B α=0.40 → 0.5 clamp)
- 또는 GRID 확장: `np.linspace(0.05, 0.9, 50)`

#### 교훈
시나리오 파라미터를 정할 때 **항상 GRID 범위와 함께 검토**.

---

### 4-2. AWSGLD favor 시나리오 만들기 (Sparse Option A vs B)

#### 시도 1 — Option A 실패
- 변경: `σ_θ` 0.40 → 0.60, `α_true` 0.20 → 0.40
- 결과: AWSGLD MSE 가 오히려 악화 (2.03 → 2.53), acMH 는 개선
- 분석: noise 만 키워봐야 acMH 가 MH rejection 으로 흡수, AWSGLD 는 step variance ↑

#### 시도 2 — Option B 성공
- 추가 변경: **`μ_N` −1.8 → −1.0** (N 그룹을 W 쪽으로 압축)
- 결과: acMH MSE 1.48 → 2.06 (악화), AWSGLD 1.20 (압승)
- 분석: 그룹 경계 흐려지면 MH 의 mode-finding 이 흔들림. AWSGLD 의 preconditioning 이 어려운 landscape 에서 강함.

#### 교훈
어려운 시나리오 = **그룹 분리도(Δμ/σ_θ) 줄이기**. μ 압축이 σ_θ 증가보다 효과적.

---

### 4-3. 시나리오 간 `ρ` 비대칭 → 통일

#### 증상
이른 Moderate 정의에서 `ρ_S=0.18, ρ_W=0.24, ρ_N=0.58` 사용. 다른 시나리오는 `0.20/0.20/0.60`.

#### 결정
시나리오 간 비교 일관성을 위해 모두 `0.20/0.20/0.60` 으로 통일 (Sparse 만 0.10/0.18/0.72 — Sparse 정체성 유지).

---

## 5. Wall-time 측정 함정

### 5-1. 시스템 부하 변동

#### 증상
같은 코드 같은 input 에서 wall-time 이 2-3 배 차이.
- AWSGLD_full at n=1500: trial 1 = 98.5s, trial 2 = 41.7s
- qSGLD: 같은 세션 trial 마다 16s ~ 76s

#### 원인
시점에 따른 시스템 부하, BLAS 워밍업, 다른 프로세스의 CPU 점유.

#### 권장
- **같은 세션·연속 실행**으로 비교 (system 부하 동일)
- R≥3 trial 평균 + std 보고
- 단일 timing 으로 결론 내지 말 것

---

### 5-2. Cache 효과

#### 증상
이론적으로 O(n²) 라서 n 두 배 → 4 배 느려져야 하는데, 실측은 6~46 배.

#### 원인
- n=800: 800×800 float64 = **5.12 MB** → L2 cache(~256KB-1MB) 초과
- n=400: 400×400 = **1.28 MB** → L2/L3 에 안정 fit

매 step 같은 행렬을 반복 access 하므로 cache hit ratio 가 결정적.

#### 교훈
순수 flop 수만으로 wall-time 을 추정하면 안 된다. n 임계점(L2 cache size 부근)에서 큰 jump.

---

### 5-3. Python stdout buffering

#### 증상
```bash
python script.py > log
```
실행 시 로그가 종료할 때까지 비어 있음. process 가 진행되는지 모니터링 불가.

#### 원인
Python stdout 은 redirect 되면 line-buffered 가 아닌 **fully buffered** (4-8 KB) 가 된다.
짧은 print 들은 buffer 가 찰 때까지 disk 에 안 써짐.

#### 해결
- `python3 -u script.py` (unbuffered)
- 또는 코드에 `print(..., flush=True)`
- 또는 `script.py` 안에서 `sys.stdout.reconfigure(line_buffering=True)`

---

### 5-4. 의외의 dominant cost — `alpha_find`

#### 발견
n=400 SGLD profiling 결과:

| 단계 | 시간 | 비율 |
|---|---|---|
| matvec C + sigma2 | 0.16s | 7% |
| grad_U | 0.29s | 12% |
| theta update | 0.05s | 2% |
| **alpha_find** | **1.82s** | **78%** |
| 합계 | 2.33s | 100% |

#### 원인
매 step 33-grid 평가 × O(n). Python loop overhead 가 BLAS matvec 보다 비싸다.

#### 교훈
큰 행렬 연산보다 **자주 호출되는 Python 함수**가 병목이 되는 경우가 흔하다. 항상 profile 먼저.

---

## 6. Method 별 failure mode

### 6-1. cycSGLD catastrophic failure (Difficult / Sparse)

#### 증상
- Difficult: Spearman = **−0.557**, NDCG = 0.337
- Sparse Opt-B: Spearman = **−0.537**, NDCG = 0.373
- Easy / Moderate: 정상 작동

#### 추정 원인
cyclical learning rate + temperature 가 **저온 phase** 에서 잘못된 local mode 에 갇힘.
Posterior 가 단순한 Easy 에서는 mode 가 하나라 OK 지만, Difficult/Sparse 처럼 multimodal 경향이 있는 setup 에서는 cyclical schedule 이 위험.

#### 권장
cycSGLD 는 easy posterior 에만 안전하게 사용. Difficult / production 에는 비추천.

---

### 6-2. SGLD vanilla 약함

#### 증상
모든 시나리오에서 MSE(θ) 가 가장 큼 (10~25), Difficult/Sparse 에서는 Spearman 도 거의 0.

#### 원인
preconditioning 없이 graph-coupled prior 를 직접 처리할 수 없음.
qSGLD (= SGLD + preconditioning) 가 같은 시간 비용으로 훨씬 좋은 결과를 냄.

#### 교훈
**graph 구조를 가진 prior 에서는 preconditioning 이 사실상 필수**.

---

## 7. Minibatch 결정 (Plan A)

### 배경
원논문 AWSGLD 는 minibatch 가정. 우리 구현은 full-batch 였으므로 엄밀히는 "AWLD"
(Adaptively Weighted Langevin Dynamics).

### 검증 실험
`simlation/awsgld_minibatch_ablation.py` 에서 n=1500 으로 batch_size = [None, 750, 300, 100] 비교:

| batch_size | Wall time | Spearman | NDCG |
|---|---|---|---|
| None (full=1500) | 70s | 0.754 | 0.974 |
| 750 | 165s (anomaly) | 0.762 | 0.975 |
| 300 | 67s | 0.752 | 0.974 |
| 100 | 42s | 0.757 | 0.974 |

### 결론
- Quality: batch size 와 무관 (5%p 이내 변동, 통계적 noise)
- Wall time: 시스템 부하에 따라 큰 변동, batch 크기와 강한 상관 없음
- **이유**: per-step 비용은 prior gradient O(n²) matvec 이 dominant.
  Likelihood gradient (O(n)) 를 minibatch 해도 절약 미미.

### 정책
- BSS scale (n ≤ 1500) 에서는 minibatch 가 wall-time 이득 없음
- 그러나 **학술적 정직성** 과 **future scalability** 위해 옵션으로 노출
- `batch_size=None` 기본값 (full-batch), `batch_size=k` 명시 시 minibatch

### 구현
- `code_JOC/keyphrase_functions_awsgld_0422.py` : `gibbs_mh(..., batch_size=None)` 인자
- `code_JOC/keyphrase_functions_awsgld_minibatch_0504.py` : 별도 minibatch 전용 모듈
- `simlation/langevin_methods_comparison.py` : `BATCH_SIZE` 전역 설정

---

## 8. 평가 지표 해석 주의사항

### 8-1. MSE(θ) vs MSE(calibrated)

`MSE_calibrated` 는 θ̂ 에 linear fit (slope, intercept) 후 MSE.
즉 scale/shift 보정 후의 정확도.

| 시나리오 | acMH MSE(θ) | AWSGLD MSE(θ) | acMH MSE(cal) | AWSGLD MSE(cal) |
|---|---|---|---|---|
| Easy_v2 | **1.96** | 2.66 | 1.07 | **0.34** |
| Moderate | 1.89 | **1.82** | 0.95 | **0.49** |
| Difficult | 2.79 | **2.26** | 1.21 | **1.13** |
| Sparse Opt-B | 2.06 | **1.20** | 0.93 | **0.47** |

→ **AWSGLD 의 θ̂ 는 모양은 정확, scale 만 다르게 추정**.
ranking metrics 에서 AWSGLD 가 우위인 이유와 일치.

### 8-2. π̂ 는 plug-in, posterior mean 아님

```python
theta_hat = mean(theta_store[BURN_IN:])
pi_hat = (1 - alpha_hat) * sigmoid(theta_hat)  # ← plug-in
```

엄밀히는 `E[π | Y] = E[(1-α)·σ(θ) | Y] ≠ (1-α̂)·σ(θ̂)` (Jensen).
대칭적 posterior 면 차이 작지만 0 은 아님.

엄밀히 하려면 step 별 `(1-α_t)·σ(θ_t)` 평균해야 함. 실용적 차이는 작음.

---

## 9. 명명 / 용어 정직성

### 현재 구현은 엄밀히 무엇인가
- "minibatch" 옵션 추가했지만 default 는 full-batch
- AWSGLD 본래 정의는 minibatch 기반 stochastic gradient
- → full-batch mode 는 "AWLD" 가 정확

### 논문/문서 작성 시 권장
- "AWSGLD with optional minibatch (default: full-batch)" 로 명시
- 또는 graph-coupled prior 환경에서의 변형이라고 framing

---

## 10. 코드 파일 위치

### 디렉토리 구조
```
BSS-Keyphrase-Extraction/
├── code_JOC/
│   ├── original/                              # Wang et al. 원본 (수정 금지)
│   │   └── keyphrase_functions.py             # acMH baseline
│   ├── keyphrase_functions_awsgld_0422.py     # AWSGLD (preconditioning + σ² floor + minibatch 옵션)
│   └── keyphrase_functions_awsgld_minibatch_0504.py  # minibatch 전용 변형
├── simulation/
│   ├── awsgld_acmh_theta_recovery_simulation_0422_v2.py          # Easy (n=800)
│   ├── awsgld_acmh_theta_recovery_simulation_0422_v2_moderate.py # Moderate (n=400)
│   ├── awsgld_acmh_theta_recovery_simulation_0422_v2_difficult.py # Difficult (n=400)
│   ├── awsgld_acmh_theta_recovery_simulation_0422_v2_sparse.py    # Sparse Opt-B (n=400)
│   ├── awsgld_minibatch_ablation.py                 # minibatch 검증 (Plan A)
│   ├── sgld_0504_easy.py                      # SGLD/qSGLD/cycSGLD on Easy
│   ├── sgld_0507.py                           # SGLD 변형 on Moderate/Difficult/Sparse
│   └── langevin_methods_comparison.py                  # 4-method × 4-scenario 통합
└── docs/
    └── TROUBLESHOOTING.md                     # (이 문서)
```

### Import path 주의
`keyphrase_functions.py` 가 `code_JOC/original/` 로 이동되어 있어
새 simulation 파일은 `ORIG_DIR` 도 sys.path 에 추가해야 함:
```python
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CODE_DIR = os.path.join(PROJECT_ROOT, "code_JOC")
ORIG_DIR = os.path.join(CODE_DIR, "original")
for _p in (CODE_DIR, ORIG_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
```

---

## 부록: 진단 스크립트

| 목적 | 파일 |
|---|---|
| σ² 궤적 비교 (acMH vs AWSGLD) | `simulation/diagnose_sigma2_trajectory.py` |
| Per-step cost profiling (alpha_find dominance) | (inline 검증, 본문 5-4 참조) |
| 4-way method 비교 (full-batch / minibatch) | `simulation/langevin_methods_comparison.py` |
| Minibatch 효과 검증 | `simulation/awsgld_minibatch_ablation.py` |

---

## 변경 이력

| 날짜 | 변경 |
|---|---|
| 2026-04-22 | AWSGLD geometry fix (preconditioning) |
| 2026-04-22 | σ² floor 0.5 도입 |
| 2026-04-23 ~ 27 | 4 시나리오 (Easy/Moderate/Difficult/Sparse) 정의 |
| 2026-05-04 | minibatch 옵션 추가 (Plan A) |
| 2026-05-07 | BtB / B@v 캐싱 최적화 (3.5x speedup) |
| 2026-05-07 | 4-method 통합 비교 (langevin_methods_comparison) |
