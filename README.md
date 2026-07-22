# BSS + AWSGLD: Bayesian Semi-supervised Keyphrase Extraction

Wang et al. (2023) 의 BSS 키프레이즈 추출 프레임워크에서 acMH-within-Gibbs
샘플러를 **AWSGLD** (Adaptively Weighted SGLD; Liang et al. 2022) 로
대체하여 다봉 사후분포에서의 trap escape 와 mode separation 을
개선한다. 비교군으로 SGLD 계열(SGLD/qSGLD/cycSGLD), 모멘텀 기반
**SGHMC** (Chen et al. 2014), MH 보정 **MALA** 를 함께 평가한다.

> **시뮬레이션 상세 보고서 → [`docs/simulation/`](docs/simulation/README.md)**
> 스터디별로 목적·데이터 생성·샘플러 설정·지표 정의·결과표·해석·재현 명령을 정리했다.
> 아래 표는 요약이며, 근거 파일과 단서 조항은 각 스터디 문서에 있다.

## 디렉토리 구조

```
docs/simulation/                   # 시뮬레이션 보고서 (본문)
├── README.md                        통합 인덱스 — 흐름/지표 정의/결과 요약/불일치 목록
├── study_1a.md  study_1b.md  study_1c.md  study_2.md  study_3.md

code_JOC/                          # BSS 모델/샘플러 코드
├── keyphrase_functions_awsgld.py    AWSGLD gibbs_mh (sigma2_floor 매개변수화)
├── mala_keyphrase.py                MALA v1/v2 (state precond + BtB 고유기저 coord step)
├── awsgld_tunable.py                eps_k 스케줄 파라미터화
├── awsgld_coordvar.py               단어별 분산 대각 preconditioner
└── original/keyphrase_functions.py  Wang et al. (2023) 원본 acMH-within-Gibbs

data_JOC/                          # Hulth·SemEval 데이터 (읽기 전용) + 전처리
├── reproduce_pos_filter.py           원논문 POS(noun/adj) 전처리 재현·문서수 검증
├── build_baseline.py                 dense(>=10)/sparse(<10) 분할 + 전처리 산출
├── extract_semeval_titles.py         SemEval 제목 추출 (관측 Y 구성용)
└── baseline_preprocessed/            전처리 결과 (pre_process/ + truth/)

simulation/
├── study_1a/                      Study 1A — 4 시나리오 × 6 sampler (난이도 sweep)
│   ├── scenario_landscapes.py       시나리오별 posterior energy landscape
│   ├── langevin_methods_comparison.py  SGLD/qSGLD/cycSGLD/SGHMC/acMH/AWSGLD
│   └── awsgld_minibatch_ablation.py    AWSGLD batch-size ablation (n=1500)
│
├── study_1b/                      Study 1B — local trap escape
│   ├── local_trap_landscape.py      3-mode mixture posterior 정의 (PARAMS)
│   ├── data_generator.py            n=400, SBM + label conflict 로 다봉 posterior 유도
│   ├── acmh_vs_awsgld.py            acMH vs AWSGLD (σ²_floor=1.0)
│   └── sgld_only.py                 SGLD / qSGLD / cycSGLD / SGHMC
│
├── study_1c/                      Study 1C — 규모(n) × 다중 seed
│   ├── data_generator.py            n ∈ {200, 1500, 10000} × seed 0–4
│   └── sampler_comparison.py        5 sampler, 3 chain pooled
│
├── study_2/                       Study 2 — 실 키프레이즈 데이터
│   ├── acmh_vs_awsgld_4to10.py      공통 하네스 (Numba acMH O(n) 증분) + 126문서 벤치
│   ├── dense_test.py                dense 실문서 macro 평가
│   ├── dense_paper_eval.py          원논문 pooled 집계 + ROC/PR 곡선
│   ├── dense_minibatch.py  dense_hp_tune.py   batch·하이퍼파라미터 ablation
│   ├── hulth_rand5_bench.py         dense 랜덤 5문서 × 10시드 기본 비교
│   ├── hulth_zeta_tune.py  hulth_eps_tune.py  ζ / ε 스케줄 스윕
│   ├── hulth_final_bench.py         acMH vs AWSGLD vs MALA_v2
│   ├── hulth_bygamma.py  hulth_weak.py        γ별 FDR / 약지도(관측 20%)
│   ├── semeval_long5.py             SemEval-2010 장문 5편
│   ├── merge_multimodal.py          문서 병합의 자연 다봉성 진단 (음성 결과)
│   ├── trap_multimode(_sharp).py    5모드 분산 트랩
│   ├── trap_consensus.py            공통키워드 + 모드별 미끼 트랩
│   ├── awsgld_convergence.py        AWSGLD vs SGHMC 4-panel 수렴 진단 (보고서 범위 밖)
│   └── archive/                     이전 중간 산출물 (근거 파일 2개만 git 추적)
│
└── study_3/                       Study 3 — 실데이터로 구성한 local trap
    ├── trap_build.py                문서별 TextRank 해를 mode 중심으로 트랩 구성·검증
    ├── trap_landscape.py  trap10_landscape.py   3문서 / 10문서 에너지 지형
    ├── trap_samplers.py             동일 타깃·온도·기하에서 acMH vs AWSGLD
    ├── tune.py  tune10.py  recall_tune.py       하이퍼파라미터 스캔
    ├── run_experiment.py            3문서 탐색·탈출
    └── run_metrics.py  run10_metrics.py         3문서 / 10문서 전 지표
```

## 비교 샘플러

| Sampler | 설명 |
|---|---|
| **acMH-within-Gibbs** | Wang et al. (2023) BSS 원본. (BᵀB)⁻¹·σ² proposal + MH 수락. |
| **SGLD** | Welling & Teh (2011) vanilla SGLD. |
| **qSGLD** | (BᵀB)⁻¹ preconditioning + Cholesky-correlated noise. |
| **cycSGLD** | Zhang et al. (2020) cyclical learning rate + 2-stage temperature. |
| **SGHMC** | Chen et al. (2014) 보조 운동량 + 마찰항으로 minibatch 노이즈 상쇄. `θ←θ+v`, `v←(1−α)v − η∇̂U + N(0,2αη)` (M=I, B̂=0). |
| **MALA v1/v2** | MH 보정으로 이산화 편향 제거. |
| **AWSGLD** | Liang et al. (2022) adaptive weighting over energy partitions + preconditioning. **본 연구 제안.** |

## Study 1A — 4 시나리오 × 6 sampler ([상세](docs/simulation/study_1a.md))

n=800, 각 시나리오 R=3 평균. 난이도는 μ/σ_θ/α/블록확률로 조절. bad init 없이 BSS 표준 초기화.
셀 = Spearman / NDCG@k.

| Sampler | Easy | Moderate | Difficult | Sparse |
|---|---|---|---|---|
| SGLD | 0.698 / 0.908 | 0.663 / 0.876 | 0.093 / 0.564 | −0.064 / 0.500 |
| qSGLD | 0.732 / 0.957 | 0.728 / 0.943 | 0.327 / 0.716 | 0.466 / 0.821 |
| cycSGLD | 0.735 / 0.958 | 0.664 / 0.904 | −0.556 / 0.337 | −0.537 / 0.373 |
| SGHMC | 0.661 / 0.891 | 0.607 / 0.852 | 0.003 / 0.538 | −0.108 / 0.498 |
| acMH | 0.727 / 0.958 | 0.705 / 0.923 | 0.059 / 0.567 | 0.125 / 0.615 |
| **AWSGLD** | **0.740 / 0.969** | **0.757 / 0.967** | **0.553 / 0.822** | **0.606 / 0.918** |

- **AWSGLD** : 4 시나리오 전부에서 Spearman·NDCG 1위. 난이도가 올라갈수록 격차 확대
  (Difficult/Sparse 에서 SGLD·cycSGLD·SGHMC 는 rank 상관이 0 또는 음수로 붕괴).
- **acMH** : Easy/Moderate 는 경쟁력 있으나 Difficult/Sparse 에서 무너지고,
  chain 당 ~700 초로 Langevin 계열의 40 배 이상 비싸다.
- **minibatch ablation** (n=1500) : batch 100 까지 줄여도 지표 변화가 셋째 자리 수준이며
  acMH 3122 초 vs AWSGLD 42 초 (74×). NDCG 는 0.974 vs 0.953 으로 AWSGLD 우위.

## Study 1B — multimodal trap escape ([상세](docs/simulation/study_1b.md))

- 데이터: n=400, p_in=0.40, p_out=0.005, damping=0.90, label conflict 48/400.
- target posterior 가 다봉임을 GD 다중 init 으로 사전 검증 (`_archive/verify_multimodal.py`).
- bad init (θ⁽⁰⁾=μ_N) 에서 3 chain × T=5000, chain 0 기준.

| Sampler | MSE_all | Spearman | NDCG@20 | R̂ max | ESS med | Cost/ESS (s) | Wall (s) |
|---|---|---|---|---|---|---|---|
| acMH | 2.282 | 0.026 | 0.518 | 1.78 | 7.50 | 22.77 | 170.8 |
| SGLD | 2.620 | 0.166 | 0.481 | 10.21 | 5.66 | 0.423 | 2.4 |
| qSGLD | 2.068 | 0.648 | **0.774** | 1.37 | 12.05 | 0.333 | 4.0 |
| cycSGLD | **1.272** | 0.692 | 0.711 | 5.09 | 4.72 | 0.841 | 4.0 |
| SGHMC | 1.704 | 0.390 | 0.665 | 4.50 | 7.09 | 0.495 | 3.5 |
| **AWSGLD** | 1.382 | **0.697** | 0.764 | **1.15** | **25.21** | **0.120** | 3.0 |

**S basin 도달 (bad init 에서 가장 먼 mode), 괄호는 median escape step**

| acMH | SGLD | qSGLD | cycSGLD | SGHMC | AWSGLD |
|---|---|---|---|---|---|
| 0/80 (—) | 0/80 (—) | 80/80 (29) | 4/80 (2484) | 15/80 (2761) | **80/80 (17)** |

- **AWSGLD** : Spearman / R̂ / ESS / Cost-per-ESS 1위. acMH 대비 cost-per-ESS 190× 효율.
- **preconditioning 이 탈출의 1차 요인, adaptive weighting 이 2차 요인.**
  (BᵀB)⁻¹ 을 쓰는 qSGLD·AWSGLD 만 S basin 완전 도달. 그 위에서 AWSGLD 가 나은 부분은
  R̂(1.15 vs 1.37)과 ESS(25.2 vs 12.0), 즉 **탈출 이후의 혼합 품질**이다.
- **acMH** : 세 그룹 θ̂ 평균이 모두 0.88~0.90 으로, 전 노드를 가운데 하나의 mode 로 뭉갠다.
- **cycSGLD** : MSE 1위지만 R̂ max 5.09 (chain 미수렴 — single-chain lucky). Study 1C 에서
  θ̂ 정의를 3 chain pooled 로 바꾸자 이 이상이 사라진다.

> ESS·Cost/ESS 는 저장된 체인과 현재 wall time 으로 재계산한 값이다. `_archive` 의 이전
> 기록(qSGLD 0.21, cycSGLD 0.51)은 더 이른 실행의 wall time 을 사용했다.

## Study 1C — 규모(n) × 다중 seed ([상세](docs/simulation/study_1c.md))

3 chain pooled posterior mean, seed 평균 (mean±std). n=10000 은 seed 0 단일. acMH 제외.

**n=200** (5 seed, NDCG@40)

| Sampler | MSE_all | Spearman | NDCG | R̂ max | Cost/ESS |
|---|---|---|---|---|---|
| SGLD | 2.742 | 0.233 | 0.618 | 5.931 | 0.704 |
| qSGLD | 1.505 | 0.763 | 0.931 | 1.660 | 0.445 |
| cycSGLD | 1.613 | **0.774** | **0.944** | 1.195 | 0.270 |
| SGHMC | 1.990 | 0.623 | 0.865 | 2.536 | 0.666 |
| **AWSGLD** | **1.005** | 0.766 | 0.943 | **1.041** | **0.062** |

**n=1500** (5 seed, NDCG@150)

| Sampler | MSE_all | Spearman | NDCG | R̂ max | Cost/ESS |
|---|---|---|---|---|---|
| SGLD | 2.716 | 0.244 | 0.620 | 6.063 | 12.32 |
| qSGLD | 1.793 | 0.753 | 0.860 | 1.765 | 7.92 |
| cycSGLD | 1.734 | 0.756 | 0.868 | 1.090 | 2.20 |
| SGHMC | 1.927 | 0.693 | 0.838 | 2.238 | 9.79 |
| **AWSGLD** | **1.232** | **0.757** | **0.874** | **1.051** | **0.899** |

**n=10000** (seed 0, NDCG@1000)

| Sampler | MSE_all | Spearman | NDCG | R̂ max | Cost/ESS |
|---|---|---|---|---|---|
| SGLD | 2.803 | 0.215 | 0.608 | 7.517 | 469.5 |
| qSGLD | 1.719 | 0.748 | 0.845 | 2.108 | 348.4 |
| cycSGLD | 1.784 | 0.754 | 0.857 | **1.088** | 81.0 |
| SGHMC | 2.084 | 0.679 | 0.838 | 2.795 | 389.3 |
| **AWSGLD** | **1.251** | **0.757** | **0.864** | **1.088** | **56.9** |

- **AWSGLD** : 모든 규모에서 MSE 1위(차선 대비 30~40% 낮음), NDCG·Spearman 최상위권,
  R̂ max ≤ 1.09, ESS 63~89 로 최고. n 이 50 배 늘어도 MSE 1.005 → 1.251 로 완만하다.
- 그룹별로 실패 지점이 갈린다. SGLD 는 MSE_S ≈ 5.1 (S basin 미도달), qSGLD 는 MSE_W 가
  규모와 함께 악화, cycSGLD·SGHMC 는 S 가 약하다. **AWSGLD 만 어느 축도 무너지지 않는다.**

## Study 2 — 실 키프레이즈 데이터 ([상세](docs/simulation/study_2.md))

Hulth 벤치마크를 원논문 방식으로 전처리(POS noun/adj, window-2 그래프)해 사용한다.
dense(키워드 ≥ 10) 391편 / sparse 109편으로 분할. acMH 는 Numba + O(n) 증분으로 가속.

### 실데이터 벤치마크

| 실험 | 설정 | 결과 |
|---|---|---|
| 126문서 (정답 4~10개) | 관측 50%, 30 sim | γ 낮으면 AWSGLD, 높으면 acMH. **AWSGLD 는 항상 더 많이 선택해 recall↑ precision↓** |
| dense 5문서 (pooled) | 관측 k=5 고정 | **AWSGLD 전 구간 우위.** ROC AUC 0.693 vs 0.664, PR 곡선도 우위 |
| dense 랜덤 5문서 | 관측 50%, 10 seed | **acMH 우위.** AUC 0.781 vs 0.761 |
| 동일 문서, 약지도 | 관측 20% | **AWSGLD 전 지표 우위.** AUC 0.676 vs 0.639, F(γ0.2) 0.623 vs 0.559 |
| MALA_v2 포함 3자 | 관측 50% | **사실상 동률** (F 0.697~0.710, AUC 0.761~0.781) |
| SemEval-2010 장문 5편 | 제목 유래 관측 | **AWSGLD 붕괴.** precision γ0.3 에서 0.121. 단 AUC 는 0.794 vs 0.802 로 대등 |

- **AWSGLD 의 우위는 관측이 희소할수록 커진다.** 라벨 신호가 약해 사후분포가 평평해질수록
  넓게 탐색하는 성질이 이득이 되고, 라벨이 충분하면 과탐색이 되어 precision 을 깎는다.
- SemEval 은 순위(AUC)가 대등한데 cutoff 만 무너지는 **캘리브레이션 문제**로 보인다.
- ζ(flat-histogram 강도) 스윕: 단봉 실데이터에서는 ζ=1 이 최적. ζ=0 은 recall 최고지만
  AUC 가 0.703 으로 급락해 순위 정보를 잃는다. ε 스케줄은 결정적 요인이 아니다.
- minibatch·τ·ζ·DECAY_LR·ε 스윕 전반에서 γ0.2 F 가 0.5965~0.6155 로 **하이퍼파라미터에 둔감**.

### 다봉성 진단 — 핵심 음성 결과

문서를 어휘 공유되게 병합하면 bridge 노드의 상충 인력으로 다봉이 생긴다는 가설을 검증했다.

| 조건 | 문서수 | n | bridge | basin 수 |
|---|---|---|---|---|
| HIGH5 (고중첩) | 5 | 314 | 70 | **1** |
| HIGH2 (최고중첩) | 2 | 113 | 41 | **1** |
| LOW5 (저중첩, 대조) | 5 | 178 | 0 | **1** |

세 에너지 정의(σ²=0.5 / 1.0 / σ²-적분) 전부 basin 1개. **가설 기각.**
실 키프레이즈 사후분포는 **단봉 heavy-tail** 이며, AWSGLD 의 mode-escape 강점은
실데이터에서 자연적으로는 발현되지 않는다. 위 벤치마크의 우위는 mode escape 가 아니라
**탐색 반경이 넓다는 성질** 자체의 효과로 해석해야 한다.

### 합성 다봉 트랩

실그래프(doc 2098)에 K=5 mode 혼합 posterior 를 주입(log-sum-exp)해 비교한다.

| 트랩 위상 | top-k | ROC AUC |
|---|---|---|
| **5모드 분산** — 정답을 모드마다 흩뿌림 | acMH 우위 (k=10 P 0.633 vs 0.533) | 0.563 vs **0.612** |
| **consensus** — 정답은 모드 공통, 미끼는 모드 고유 | **AWSGLD 저·중 k 우위** (k=5 P **0.000 vs 0.600**) | 0.571 vs **0.668** |

- 분산형에서는 **커버리지 vs 선명도 트레이드오프** 때문에 탈출해도 이득이 없다.
  탈출에 필요한 온도가 곧 π̂ 가 흐려지는 온도다.
- consensus 형에서는 acMH 가 갇힌 모드의 미끼에 오염돼 top-5 precision 0.000,
  FDR-cutoff 전 γ 에서 실현 FDR 1.000 (선택한 것이 전부 미끼). AWSGLD 는 여러 모드를
  평균내 미끼를 상쇄하고 공통 키워드만 부상시킨다.

## Study 3 — 실데이터로 구성한 local trap ([상세](docs/simulation/study_3.md))

Study 2 의 합성 트랩은 mode 중심을 사람이 지정했다는 한계가 있었다. Study 3 은 중심을
**문서별 TextRank 해**로 두어 지형 좌표를 전부 실데이터 산출값으로 만든다.

- Hulth 문서 K편을 합집합 어휘로 병합(같은 stem = 같은 노드), 공유 그래프 B 계산
- 중심 u^(k) = 문서 k 자체 그래프의 TextRank 해, `U_mix = −logsumexp(−U_k)`
- 두 샘플러가 **같은 타깃·같은 온도·같은 (BᵀB)⁻¹ 기하**를 쓰므로 차이는 flat-histogram 유무뿐
- 3문서: n=118, basin 3, 장벽 29.9(깊음) / 7.9(얕음) — 10시드, T=12000
- 10문서: n=322, basin 10, 장벽 3.3~22.5 — 10시드, T=20000

**탐색**

| | 방문 basin | 탈출율 |
|---|---|---|
| acMH (3문서) | 1.0 / 3 | **0.00** |
| AWSGLD (3문서) | **2.2 ± 0.75** | **0.80** |
| acMH (10문서) | 1.10 / 10 | 0.20 |
| AWSGLD (10문서) | **7.40 ± 2.06** | **1.00** |

**성능** (10시드 평균)

| 문서 | Sampler | P@20 | ROC AUC | NDCG@20 | 실현 FDR (γ=0.20) |
|---|---|---|---|---|---|
| 3문서 | acMH | 0.555 | 0.607 | 0.661 | 0.306 |
| 3문서 | **AWSGLD** | **0.745** | **0.712** | **0.806** | **0.084** |
| 10문서 | acMH | 0.810 | 0.692 | 0.833 | — |
| 10문서 | **AWSGLD** | **0.945** | **0.722** | **0.944** | — |

- **acMH 의 갇힘은 완전하다.** 3문서 10시드 전부에서 출발 basin 체류 비율이 정확히 1.000.
  Study 1B 의 합성 실험 결론을 **실데이터 지형에서 재현**한 것이다.
- **선택 개수 방향이 뒤집힌 것이 핵심 관찰.** 단봉 실데이터(Study 2)에서 AWSGLD 는 과선택으로
  FDR 을 위반했는데, 다봉 지형에서는 보수적으로 선택하고 명목 FDR 을 지킨다. acMH 가 오히려
  갇힌 basin 의 비키워드에 확신을 부여해 FDR 을 크게 위반한다.
- 한계: 중심은 실데이터 산출값이지만 `BASELINE=−2.0`, `σ²=0.5` 는 다봉 형성을 위해 고른 값이다
  (초기 스캔 `BASELINE=−0.5` 에서는 basin 이 뭉개졌다). 인위성이 완전히 제거되지는 않았다.

## 지표 정의

- **MSE_all** : 전체 노드 점추정 MSE (logit space)
- **Spearman** : θ̂ vs θ\* rank 상관
- **NDCG@k** : rank 기반 graded relevance 의 top-k ranking quality (∈[0,1])
- **R̂ max** : Gelman-Rubin 노드별 최대 (< 1.1 양호, > 1.2 수렴 실패)
- **ESS / Cost per ESS** : Geyer initial positive sequence 기준 / wall_time ÷ ESS median
- **ROC AUC** : cutoff 선택과 무관해 트랩 실험에서 가장 신뢰도가 높다
- **FDR-cutoff P/R/F** : 명목 FDR γ 에서 자른 성능 + 실현 FDR.
  `force_obs_to_key2` 가 Y=1 을 π=1 로 고정하므로 트랩 실험에서는 precision=1.0 인공물에 주의

## 재현

```bash
# Study 1A — 4 시나리오 × sampler (acMH 포함 시 ~700s/trial 로 매우 느림)
python3 simulation/study_1a/langevin_methods_comparison.py
python3 simulation/study_1a/scenario_landscapes.py

# Study 1B — local trap escape
python3 simulation/study_1b/data_generator.py 0          # 데이터 생성 (seed 0)
python3 simulation/study_1b/acmh_vs_awsgld.py 0          # acMH + AWSGLD (~10 분)
python3 simulation/study_1b/sgld_only.py 0               # SGLD/qSGLD/cycSGLD/SGHMC (~15 초)

# Study 1C — 규모 × 다중 seed
python3 simulation/study_1c/data_generator.py --n 200 --seed 0
python3 simulation/study_1c/sampler_comparison.py --n 200        # n=1500/10000 도 가능

# Study 2 — 전처리 → 실데이터 벤치마크
python3 data_JOC/build_baseline.py                       # dense/sparse baseline 생성
python3 simulation/study_2/dense_paper_eval.py           # 원논문 pooled 집계 + ROC/PR
python3 simulation/study_2/hulth_weak.py                 # 약지도(관측 20%)
python3 simulation/study_2/semeval_long5.py              # SemEval 장문 (제목 추출 선행)
python3 simulation/study_2/merge_multimodal.py           # 다봉성 진단 (음성 결과)
python3 simulation/study_2/trap_consensus.py             # consensus 트랩

# Study 3 — 실데이터 local trap
python3 simulation/study_3/trap_landscape.py             # 3문서 지형
python3 simulation/study_3/run_experiment.py             # 탐색·탈출 (~45 초)
python3 simulation/study_3/run_metrics.py                # 3문서 전 지표 (표만 출력)
python3 simulation/study_3/run10_metrics.py              # 10문서 전 지표 (~290 초)
```

`.npz` chain 결과 파일은 `.gitignore` 로 제외 (대용량, `study_1c/` 만 약 7.7 GB). 재실행으로 재생성된다.
`simulation/study_2/archive/` 는 폐기된 중간 산출물이라 문서가 인용하는 근거 파일 2개만 추적한다.

## 진행 상황

- [x] Wang et al. (2023) BSS 원본 코드 분석
- [x] AWSGLD 변형 구현 (`code_JOC/keyphrase_functions_awsgld.py`)
- [x] Study 1A : 4 시나리오 × 6 sampler (Easy/Moderate/Difficult/Sparse) + minibatch ablation
- [x] Study 1B : multimodal trap escape 에서 6 sampler 비교
- [x] Study 1C : 규모(200/1500/10000) × 다 seed, 3 chain pooled 추정
- [x] AWSGLD σ²_floor 튜닝 (sweep → 1.0 fix)
- [x] 평가 지표 다축화 (MSE / Spearman / NDCG@k / ESS / R̂ / ROC AUC)
- [x] SGHMC (Chen et al. 2014) baseline 통합
- [x] MALA v1/v2 샘플러 구현 및 비교
- [x] Hulth real 키프레이즈 데이터 전처리(POS) + baseline (dense/sparse)
- [x] Study 2 : 실문서 acMH vs AWSGLD (dense/약지도 → AWSGLD, 충분관측 → acMH)
- [x] Study 2 : 원논문 pooled 집계 + ROC/PR 재평가
- [x] Study 2 : SemEval-2010 장문 확장
- [x] Study 2 : 문서 병합의 자연 다봉성 진단 → **단봉 확인 (가설 기각)**
- [x] Study 2 : 합성 다봉 트랩 (consensus) 에서 mode-escape 우위 입증
- [x] Study 3 : 실데이터 기반 local trap 구성 및 acMH vs AWSGLD 검증
- [x] 시뮬레이션 보고서 정리 (`docs/simulation/`)
- [ ] Study 3 결과를 `simulation/BSS_032926.tex` 에 반영
- [ ] SGHMC 하이퍼파라미터 (lr/friction) 튜닝
- [ ] Study 3 트랩의 `BASELINE` / `σ²` 민감도 분석
- [ ] SemEval 장문의 FDR-cutoff 캘리브레이션 붕괴 원인 규명
- [ ] 스크립트의 `/home/jiyoon` 절대경로 하드코딩 제거 (재현성)

## 참고

- Wang et al. (2023), *Bayesian Semi-supervised Keyphrase Extraction*, INFORMS Journal on Computing.
- Liang et al. (2022), *Adaptively Weighted Stochastic Gradient MCMC*.
- Chen et al. (2014), *Stochastic Gradient Hamiltonian Monte Carlo*, ICML.
- Welling & Teh (2011), *Bayesian Learning via Stochastic Gradient Langevin Dynamics*, ICML.
- Zhang et al. (2020), *Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning*, ICLR.
