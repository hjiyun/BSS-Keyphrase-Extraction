# BSS + AWSGLD: Bayesian Semi-supervised Keyphrase Extraction

Wang et al. (2023) 의 BSS 키프레이즈 추출 프레임워크에서 acMH-within-Gibbs
샘플러를 **AWSGLD** (Adaptively Weighted SGLD; Liang et al. 2022) 로
대체하여 multimodal 사후분포에서의 trap escape 와 mode separation 을
개선한다.

## 디렉토리 구조

```
code_JOC/                          # BSS 모델/샘플러 코드
├── keyphrase_functions_awsgld.py    AWSGLD gibbs_mh (sigma2_floor 매개변수화)
└── original/
    └── keyphrase_functions.py       Wang et al. (2023) 원본 acMH-within-Gibbs

simulation/
├── study_1a/                      Study 1A — 4 시나리오 × 5 샘플러 비교
│   ├── scenario_landscapes.py       4 시나리오 (Easy/Moderate/Difficult/Sparse) energy landscape
│   ├── langevin_methods_comparison.py  acMH + SGLD/qSGLD/cycSGLD + AWSGLD 비교
│   ├── awsgld_minibatch_ablation.py    AWSGLD batch-size ablation
│   └── _archive/                    이전 실험 보관
│
└── study_1b/                      Study 1B — local trap escape + 5 sampler 비교
    ├── local_trap_landscape.py      Target 3-mode mixture posterior 정의 (PARAMS)
    ├── data_generator.py            n=400, SBM + label conflict 로 multimodal BSS posterior 유도
    ├── data_landscape_overview.py   target energy + θ* truth histogram 시각화
    ├── acmh_vs_awsgld.py            acMH vs AWSGLD (σ²_floor=1.0)
    ├── sgld_only.py                 SGLD / qSGLD / cycSGLD 3종
    └── _archive/                    σ² sweep, multimodal 검증 (GD), post-hoc metric 보조 스크립트
```

## 비교 샘플러 (5 종)

| Sampler | 설명 |
|---|---|
| **acMH-within-Gibbs** | Wang et al. (2023) BSS 원본. (BᵀB)⁻¹·σ² proposal + MH 수락. |
| **SGLD** | Welling & Teh (2011) vanilla SGLD. |
| **qSGLD** | (BᵀB)⁻¹ preconditioning + Cholesky-correlated noise. |
| **cycSGLD** | Zhang et al. (2020) cyclical learning rate + 2-stage temperature. |
| **AWSGLD** | Liang et al. (2022) adaptive weighting over energy partitions + preconditioning. **본 연구 제안.** |

## Study 1B 핵심 결과 (multimodal trap setup)

- 데이터: n=400, p_in=0.40, p_out=0.005, damping=0.90, label conflict 48/400.
- target posterior 가 multimodal 임을 GD 다중 init 으로 사전 검증 (이전 `_archive/verify_multimodal.py`).
- bad init (θ⁽⁰⁾=μ_N) 에서 3 chain × T=5000 으로 비교.

| Sampler | MSE_all | Spearman | R̂ max | Cost/ESS (s) | Wall (s) |
|---|---|---|---|---|---|
| acMH | 2.282 | 0.026 | 1.78 | 22.77 | 170.8 |
| SGLD | 2.620 | 0.166 | 10.21 | 0.42 | 2.4 |
| qSGLD | 2.068 | 0.648 | 1.37 | 0.21 | 2.6 |
| cycSGLD | **1.272** | 0.692 | 5.09 | 0.51 | 2.4 |
| **AWSGLD** | 1.382 | **0.697** | **1.15** | **0.12** | 3.0 |

- **AWSGLD** : Spearman / R̂ / Cost-per-ESS 3 metric 1위. acMH 대비 cost-per-ESS 190× 효율.
- **cycSGLD** : MSE 1위 BUT R̂ max 5.09 (chain 미수렴 — single-chain lucky 결과).

지표 정의:
- **MSE_all** : 전체 노드 점추정 MSE
- **Spearman** : θ̂ vs θ* rank 상관
- **R̂ max** : Gelman-Rubin 노드별 최대 (< 1.1 양호, > 1.2 수렴 실패)
- **Cost/ESS** : wall_time / 효과적 독립 sample 수 (시간당 정보 비용)

## 재현

```bash
# Study 1A — 4 시나리오 × 5 sampler
python3 simulation/study_1a/langevin_methods_comparison.py
python3 simulation/study_1a/scenario_landscapes.py

# Study 1B — local trap escape
python3 simulation/study_1b/data_generator.py 0          # 데이터 생성 (seed 0)
python3 simulation/study_1b/data_landscape_overview.py 0 # 분포 시각화
python3 simulation/study_1b/acmh_vs_awsgld.py 0          # acMH + AWSGLD (~10 분)
python3 simulation/study_1b/sgld_only.py 0               # SGLD/qSGLD/cycSGLD (~10 초)
```

`.npz` chain 결과 파일은 `.gitignore` 로 제외 (대용량). 재실행으로 재생성됨.

## 진행 상황

- [x] Wang et al. (2023) BSS 원본 코드 분석
- [x] AWSGLD 변형 구현 (`code_JOC/keyphrase_functions_awsgld.py`)
- [x] Study 1A : 4 시나리오 × 5 sampler 비교 (Easy/Moderate/Difficult/Sparse)
- [x] Study 1B : multimodal trap setup (label conflict + 강한 graph) 에서 5 sampler 비교
- [x] AWSGLD σ²_floor 튜닝 (sweep → 1.0 fix)
- [x] 평가 지표 다축화 (MSE / Spearman / NDCG@k / ESS / R̂)
- [ ] Study 1C : mini-batch / 큰 n / 다 seed
- [ ] Hulth 등 real 키프레이즈 데이터 평가

## 참고

- Wang et al. (2023), *Bayesian Semi-supervised Keyphrase Extraction*, INFORMS Journal on Computing.
- Liang et al. (2022), *Adaptively Weighted Stochastic Gradient MCMC*.
