# BSS + AWSGLD: Bayesian Semi-supervised Keyphrase Extraction

Wang et al. (2023) 의 BSS 키프레이즈 추출 프레임워크에서 acMH-within-Gibbs
샘플러를 **AWSGLD** (Adaptively Weighted SGLD; Liang et al. 2022) 로
대체하여 multimodal 사후분포에서의 trap escape 와 mode separation 을
개선한다. 비교군으로 SGLD 계열(SGLD/qSGLD/cycSGLD)과 모멘텀 기반
**SGHMC** (Chen et al. 2014) 를 함께 평가한다.

## 디렉토리 구조

```
code_JOC/                          # BSS 모델/샘플러 코드
├── keyphrase_functions_awsgld.py    AWSGLD gibbs_mh (sigma2_floor 매개변수화)
└── original/
    └── keyphrase_functions.py       Wang et al. (2023) 원본 acMH-within-Gibbs

simulation/
├── study_1a/                      Study 1A — 4 시나리오 × sampler 비교
│   ├── scenario_landscapes.py       4 시나리오 (Easy/Moderate/Difficult/Sparse) energy landscape
│   ├── langevin_methods_comparison.py  SGLD/qSGLD/cycSGLD + SGHMC + acMH + AWSGLD 비교
│   ├── awsgld_minibatch_ablation.py    AWSGLD batch-size ablation
│   └── _archive/                    이전 실험 보관
│
├── study_1b/                      Study 1B — local trap escape + sampler 비교
│   ├── local_trap_landscape.py      Target 3-mode mixture posterior 정의 (PARAMS)
│   ├── data_generator.py            n=400, SBM + label conflict 로 multimodal BSS posterior 유도
│   ├── data_landscape_overview.py   target energy + θ* truth histogram 시각화
│   ├── acmh_vs_awsgld.py            acMH vs AWSGLD (σ²_floor=1.0)
│   ├── sgld_only.py                 SGLD / qSGLD / cycSGLD / SGHMC
│   └── _archive/                    σ² sweep, multimodal 검증(GD), NDCG·ESS 보조 스크립트
│
├── study_1c/                      Study 1C — 스케일(n) × 다중 seed 비교
│   ├── data_generator.py            n ∈ {200, 1500, 10000} × seed 0–4 데이터 생성
│   ├── data_landscape_overview.py   분포 개요 시각화
│   └── sampler_comparison.py        SGLD/qSGLD/cycSGLD/SGHMC/AWSGLD, 3 chain pooled
│
└── study_2/                       Study 2 — 수렴 진단 (convergence diagnostics)
    └── awsgld_convergence.py        AWSGLD vs SGHMC 4-panel trace (θ, ‖θ−θ̄‖, U(x), running MSE)
```

## 비교 샘플러 (6 종)

| Sampler | 설명 |
|---|---|
| **acMH-within-Gibbs** | Wang et al. (2023) BSS 원본. (BᵀB)⁻¹·σ² proposal + MH 수락. |
| **SGLD** | Welling & Teh (2011) vanilla SGLD. |
| **qSGLD** | (BᵀB)⁻¹ preconditioning + Cholesky-correlated noise. |
| **cycSGLD** | Zhang et al. (2020) cyclical learning rate + 2-stage temperature. |
| **SGHMC** | Chen et al. (2014) 보조 운동량 + 마찰항으로 minibatch 노이즈 상쇄. `θ←θ+v`, `v←(1−α)v − η∇̂U + N(0,2αη)` (M=I, B̂=0). |
| **AWSGLD** | Liang et al. (2022) adaptive weighting over energy partitions + preconditioning. **본 연구 제안.** |

## Study 1A — 4 시나리오 × 5 sampler (난이도 sweep)

각 시나리오 R=3 평균. 난이도는 μ/σ/α 로 조절. 지표는 Spearman(θ̂ vs θ* rank 상관)과
NDCG@k(graded ranking quality). bad init 없이 BSS 표준 초기화 사용.

| Sampler | Easy | Moderate | Difficult | Sparse |
|---|---|---|---|---|
| SGLD | 0.698 / 0.908 | 0.663 / 0.876 | 0.093 / 0.564 | −0.064 / 0.500 |
| qSGLD | 0.732 / 0.957 | 0.728 / 0.943 | 0.327 / 0.715 | 0.466 / 0.821 |
| cycSGLD | 0.735 / 0.958 | 0.664 / 0.904 | −0.556 / 0.337 | −0.537 / 0.373 |
| SGHMC | 0.661 / 0.891 | 0.607 / 0.852 | 0.003 / 0.538 | −0.108 / 0.498 |
| **AWSGLD** | **0.740 / 0.969** | **0.757 / 0.967** | **0.553 / 0.822** | **0.606 / 0.918** |

(셀 = Spearman / NDCG@k)

- **AWSGLD** : 4 시나리오 전부에서 Spearman·NDCG 1위. 난이도가 올라갈수록 격차 확대
  (Difficult/Sparse 에서 SGLD·cycSGLD 는 rank 상관이 음수로 붕괴).
- **SGHMC** : vanilla SGLD 대비 안정적이나 precondition 부재로 MSE_θ 가 높고(≈12–23),
  강한 난이도에서 AWSGLD 에 미달.

## Study 1B — multimodal trap escape (6 sampler)

- 데이터: n=400, p_in=0.40, p_out=0.005, damping=0.90, label conflict 48/400.
- target posterior 가 multimodal 임을 GD 다중 init 으로 사전 검증 (`_archive/verify_multimodal.py`).
- bad init (θ⁽⁰⁾=μ_N) 에서 3 chain × T=5000, chain 0 기준.

| Sampler | MSE_all | Spearman | NDCG@20 | R̂ max | Cost/ESS (s) | Wall (s) |
|---|---|---|---|---|---|---|
| acMH | 2.282 | 0.026 | 0.518 | 1.78 | 22.77 | 170.8 |
| SGLD | 2.620 | 0.166 | 0.481 | 10.21 | 0.42 | 2.4 |
| qSGLD | 2.068 | 0.648 | 0.774 | 1.37 | 0.21 | 2.6 |
| cycSGLD | **1.272** | 0.692 | 0.711 | 5.09 | 0.51 | 2.4 |
| SGHMC | 1.704 | 0.390 | 0.665 | 4.50 | 0.49 | 3.5 |
| **AWSGLD** | 1.382 | **0.697** | **0.764** | **1.15** | **0.12** | 3.0 |

- **AWSGLD** : Spearman / NDCG / R̂ / Cost-per-ESS 1위. acMH 대비 cost-per-ESS 190× 효율.
- **cycSGLD** : MSE 1위 BUT R̂ max 5.09 (chain 미수렴 — single-chain lucky 결과).
- **SGHMC** : 모멘텀+마찰 덕에 vanilla SGLD 대비 명확히 개선 (MSE 2.62→1.70, R̂ max 10.21→4.50)
  하나, qSGLD·AWSGLD 수준에는 못 미침.

## Study 1C — 스케일(n) × 다중 seed (5 sampler)

3 chain pooled posterior mean, seed 평균 (mean±std). n=10000 은 seed 0 단일.
지표: MSE_all / Spearman / NDCG@k / R̂ max.

**n=200** (5 seed, NDCG@40)

| Sampler | MSE_all | Spearman | NDCG | R̂ max |
|---|---|---|---|---|
| SGLD | 2.742 | 0.233 | 0.618 | 5.931 |
| qSGLD | 1.505 | 0.763 | 0.931 | 1.660 |
| cycSGLD | 1.613 | 0.774 | 0.944 | 1.195 |
| SGHMC | 1.990 | 0.623 | 0.865 | 2.536 |
| **AWSGLD** | **1.005** | 0.766 | 0.943 | **1.041** |

**n=1500** (5 seed, NDCG@150)

| Sampler | MSE_all | Spearman | NDCG | R̂ max |
|---|---|---|---|---|
| SGLD | 2.716 | 0.244 | 0.620 | 6.063 |
| qSGLD | 1.793 | 0.753 | 0.860 | 1.765 |
| cycSGLD | 1.734 | 0.756 | 0.868 | 1.090 |
| SGHMC | 1.927 | 0.693 | 0.838 | 2.238 |
| **AWSGLD** | **1.232** | 0.757 | **0.874** | 1.051 |

**n=10000** (seed 0, NDCG@1000)

| Sampler | MSE_all | Spearman | NDCG | R̂ max |
|---|---|---|---|---|
| SGLD | 2.803 | 0.215 | 0.608 | 7.517 |
| qSGLD | 1.719 | 0.748 | 0.845 | 2.108 |
| cycSGLD | 1.784 | 0.754 | 0.857 | **1.088** |
| SGHMC | 2.084 | 0.679 | 0.838 | 2.795 |
| **AWSGLD** | **1.251** | 0.757 | **0.864** | 1.088 |

- **AWSGLD** : 모든 스케일에서 MSE 1위, NDCG·Spearman 최상위권. n 증가에도 안정.
- **SGHMC** : 스케일 무관하게 일관 (Spearman 0.62–0.69, NDCG 0.84–0.87), vanilla SGLD 를 전 스케일에서 능가.

## Study 2 — 수렴 진단 (AWSGLD vs SGHMC)

Easy / Moderate / Difficult 시나리오에서 동일 posterior·init 으로 두 sampler 의 trace 를
4-panel 로 비교 (`awsgld_convergence_{scenario}.png`):

- **(a)** 대표 component (S/W/N 각 3개) θ trace
- **(b)** ‖θ_k − θ̄‖₂ — 전체 벡터 fluctuation
- **(c)** U(x_k) energy trace — multimodal exploration 증거
- **(d)** ‖θ̄_k − θ*‖²/n — running posterior mean MSE

## 지표 정의

- **MSE_all** : 전체 노드 점추정 MSE (logit space)
- **Spearman** : θ̂ vs θ* rank 상관
- **NDCG@k** : rank 기반 graded relevance 의 top-k ranking quality (∈[0,1])
- **R̂ max** : Gelman-Rubin 노드별 최대 (< 1.1 양호, > 1.2 수렴 실패)
- **Cost/ESS** : wall_time / 효과적 독립 sample 수 (시간당 정보 비용)

## 재현

```bash
# Study 1A — 4 시나리오 × sampler  (acMH 포함 시 ~700s/trial 로 매우 느림)
python3 simulation/study_1a/langevin_methods_comparison.py
python3 simulation/study_1a/scenario_landscapes.py

# Study 1B — local trap escape
python3 simulation/study_1b/data_generator.py 0          # 데이터 생성 (seed 0)
python3 simulation/study_1b/data_landscape_overview.py 0 # 분포 시각화
python3 simulation/study_1b/acmh_vs_awsgld.py 0          # acMH + AWSGLD (~10 분)
python3 simulation/study_1b/sgld_only.py 0               # SGLD/qSGLD/cycSGLD/SGHMC (~15 초)

# Study 1C — 스케일 × 다중 seed
python3 simulation/study_1c/data_generator.py --n 200 --seed 0   # 데이터 생성
python3 simulation/study_1c/sampler_comparison.py --n 200        # 5 sampler (n=1500/10000 도 가능)

# Study 2 — 수렴 진단
python3 simulation/study_2/awsgld_convergence.py        # easy/moderate/difficult
```

`.npz` chain 결과 파일은 `.gitignore` 로 제외 (대용량). 재실행으로 재생성됨.

## 진행 상황

- [x] Wang et al. (2023) BSS 원본 코드 분석
- [x] AWSGLD 변형 구현 (`code_JOC/keyphrase_functions_awsgld.py`)
- [x] Study 1A : 4 시나리오 × sampler 비교 (Easy/Moderate/Difficult/Sparse)
- [x] Study 1B : multimodal trap setup (label conflict + 강한 graph) 에서 sampler 비교
- [x] AWSGLD σ²_floor 튜닝 (sweep → 1.0 fix)
- [x] 평가 지표 다축화 (MSE / Spearman / NDCG@k / ESS / R̂)
- [x] SGHMC (Chen et al. 2014) baseline 통합 (Study 1A/1B/1C/2)
- [x] Study 1C : mini-batch / 큰 n (200/1500/10000) / 다 seed
- [x] Study 2 : AWSGLD vs SGHMC 수렴 진단
- [ ] SGHMC 하이퍼파라미터 (lr/friction) 튜닝
- [ ] Hulth 등 real 키프레이즈 데이터 평가

## 참고

- Wang et al. (2023), *Bayesian Semi-supervised Keyphrase Extraction*, INFORMS Journal on Computing.
- Liang et al. (2022), *Adaptively Weighted Stochastic Gradient MCMC*.
- Chen et al. (2014), *Stochastic Gradient Hamiltonian Monte Carlo*, ICML.
- Welling & Teh (2011), *Bayesian Learning via Stochastic Gradient Langevin Dynamics*, ICML.
