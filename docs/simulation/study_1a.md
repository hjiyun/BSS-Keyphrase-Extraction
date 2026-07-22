# Study 1A — 시나리오 난이도 sweep에서의 샘플러 비교

> 산출물 위치: `simulation/study_1a/`
> 결과 근거 파일: `langevin_methods_comparison_with_acmh_summary.json`, `awsgld_minibatch_ablation_summary.json`, `scenario_landscape_*.png`

## 1. 목적

난이도를 통제한 합성 데이터 4종에서 6개 샘플러의 잠재점수 θ 복원 성능을 비교한다.
난이도는 그룹 평균 μ의 분리도, 그룹 내 분산 σ_θ, PU 라벨잡음 α, 그래프 블록 확률로만
조절하며 샘플러에게 주는 입력(그래프·라벨·초기값)은 전부 동일하게 맞춘다.

핵심 질문은 "난이도가 올라갈 때 어떤 샘플러가 먼저 무너지는가"이다.

## 2. 데이터 생성 (DGP)

n = 800, 3-block SBM. 시드 base = 20260507, 시나리오당 R = 3회 반복.

1. **그룹 배정** — z_i ∈ {S, W, N}을 비율 ρ = (ρ_S, ρ_W, ρ_N)만큼 정확히 만들고 permutation한다.
   S = strong keyphrase, W = weak keyphrase, N = non-keyphrase.
2. **그래프 A** — 블록 확률로 상삼각을 Bernoulli 샘플링 후 대칭화한다.
   같은 그룹 = `within`, S–W 쌍 = `between_sw`, 그 외 = `between_other`.
   고립 노드는 임의 노드와 연결해 B가 잘 정의되도록 보정한다.
3. **참값 θ\*** — θ\*_i = μ_{z_i} + ε_i, ε_i ~ N(0, σ_θ²).
4. **관측 라벨 Y** — Y_i ~ Bernoulli((1 − α\*)·sigmoid(θ\*_i)).
   즉 관측확률은 π_i 자체가 아니라 PU 잡음이 곱해진 (1 − α\*)π_i이다.
5. **BSS 입력** (d = 0.85)
   - D = diag(deg(A)), B = I − d·(D⁻¹A)ᵀ, B\* = I − d·D^(−1/2) A D^(−1/2)
   - u_0 = B⁻¹(1 − d)**1** ← prior mean (**oracle 초기화 아님**)
   - ini = `base_to_start`(B\*⁻¹Y) ← BSS 원본 초기값
   - α̂⁽⁰⁾ = `alpha_find`(u_0, Y, grid), grid = (10…42 − 5)/(10…42)

### 시나리오 파라미터

| 시나리오 | ρ (S/W/N) | μ_S / μ_W / μ_N | σ_θ | α\* | block (within / sw / other) | 평균 관측수 |
|---|---|---|---|---|---|---|
| Easy | 0.20 / 0.20 / 0.60 | 2.5 / 1.0 / −2.5 | 0.35 | 0.20 | 0.20 / 0.03 / 0.005 | 243.7 |
| Moderate | 0.20 / 0.20 / 0.60 | 2.0 / 0.5 / −1.8 | 0.50 | 0.35 | 0.20 / 0.03 / 0.005 | 196.7 |
| Difficult | 0.20 / 0.20 / 0.60 | 1.5 / 0.0 / −1.0 | 0.60 | 0.50 | 0.15 / 0.05 / 0.01 | 171.0 |
| Sparse | 0.10 / 0.18 / 0.72 | 2.0 / 1.0 / −1.0 | 0.55 | 0.40 | 0.20 / 0.03 / 0.005 | 190.0 |

난이도 상승 요인은 세 가지가 동시에 작동한다. (a) μ 간격 축소 + σ_θ 증가로 세 mode가
겹치고, (b) α 증가로 Y=1 신호가 희석되며, (c) Difficult는 블록 대비까지 약화된다.
Sparse는 키프레이즈 비율 자체를 0.40 → 0.28로 낮춘 불균형 조건이다.

### 타깃 분포

샘플러가 겨냥하는 사후 에너지는 다음과 같다.

```
U(θ) = −log p(Y | θ, α) + ‖B(θ − u_0)‖² / (2σ²)
σ² ~ InvGamma(n/2 + 0.001, C/2 + 0.001),  C = ‖B(θ − u_0)‖²  (매 스텝 Gibbs)
σ² ← max(σ², σ²_floor),  floor = 0.5
```

`scenario_landscapes.py`는 이 사후분포를 시나리오별 1D mixture 형태로 시각화한다.
CLAUDE.md 규칙에 따라 가중합이 아니라 log-sum-exp mixture로 계산한다.

```
E(θ) = −log[ ρ_S e^(−E_S(θ)) + ρ_W e^(−E_W(θ)) + ρ_N e^(−E_N(θ)) ]
E_S, E_W (Y=1) = −log[(1−α)·sigmoid(θ)] + (θ−μ)²/(2σ_θ²)
E_N     (Y=0) = −log[1 − (1−α)·sigmoid(θ)] + (θ−μ)²/(2σ_θ²)
```

## 3. 샘플러 설정

T = 5000, burn-in = 1000, minibatch = 100 (likelihood gradient만; prior 항 B ᵀB는 항상 full-batch).
모든 샘플러가 매 스텝 σ²를 Gibbs로 갱신하고 α̂를 `alpha_find`로 재추정한다.

| 샘플러 | 갱신식 | 하이퍼파라미터 |
|---|---|---|
| SGLD | θ ← θ − ε_k ∇U + √(2τε_k)·N(0,I) | ε_k = 0.02/((t+1)^0.6 + 10), τ = 1 |
| qSGLD | θ ← θ − ε_k P∇U + √(2τε_k)·L·N(0,I), LLᵀ = P ≈ (BᵀB)⁻¹ | ε_k = 0.3/((t+1)^0.6 + 10) |
| cycSGLD | 순환 학습률 + 2단계 온도 (탐색 τ/1e4 → 샘플링 τ) | ε_base = 0.01, 10 cycle |
| SGHMC | θ ← θ + v, v ← (1−a)v − η∇̂U + N(0, 2aη) (M = I, B̂ = 0) | η_base = 0.01, friction = 0.1 |
| AWSGLD | qSGLD 구조 + 에너지 분할 위 adaptive weight로 gradient 배율 조정 | ε_k = 0.3/((t+1)^0.6+10), τ = 1, ζ = 5, M = 1000 구간, σ²_floor = 0.5 |
| acMH | 원본 BSS. 성분별 MH, proposal cov ∝ (BᵀB)⁻¹σ²·4/n | 원본 `keyphrase_functions.gibbs_mh` |

AWSGLD의 gradient 배율은 warm-up 100 스텝으로 에너지 범위를 잡은 뒤
`grad_mult = 1 + (ζτ/Δu)·[log w_J − log w_{J−1}]`를 [0.1, 10]으로 clip해서 적용한다.
방문한 에너지 구간의 가중치를 키워 그 구간을 눌러 평탄화하는 flat-histogram 기제이다.

## 4. 평가 지표

θ̂ = post-burn 표본평균, π̂ = (1 − α̂)·sigmoid(θ̂).

- **MSE_θ** — mean((θ̂ − θ\*)²). 스케일 왜곡에 민감하다.
- **MSE_cal** — θ̂를 θ\*에 1차 회귀로 보정한 뒤의 MSE. 순서는 맞는데 스케일만 어긋난 경우를 분리한다.
- **Spearman / Kendall** — θ̂ vs θ\* 순위 상관.
- **top-k overlap** — k = #{θ\*_i > 0}, 상위 k 집합의 교집합 비율.
- **NDCG@k** — 순위 기반 graded relevance(θ\* 순위를 [0,1]로 정규화)로 계산한 상위 k 품질.
- **wall time** — chain 1개 소요 시간(초).

## 5. 결과

각 셀은 R = 3 평균이다.

### Easy

| 샘플러 | MSE_θ | MSE_cal | Spearman | Kendall | top-k | NDCG@k | α̂ | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 13.447 | 1.612 | 0.698 | 0.465 | 0.919 | 0.908 | 0.623 | 16.8 |
| qSGLD | 1.826 | 0.988 | 0.732 | 0.502 | 0.980 | 0.957 | 0.500 | 16.1 |
| cycSGLD | 6.925 | 0.643 | 0.735 | 0.502 | 0.995 | 0.958 | 0.569 | 16.0 |
| SGHMC | 12.070 | 2.002 | 0.661 | 0.437 | 0.887 | 0.891 | 0.603 | 17.2 |
| **AWSGLD** | 2.734 | **0.377** | **0.740** | **0.513** | **0.999** | **0.969** | 0.500 | 16.5 |
| acMH | **2.149** | 1.039 | 0.727 | 0.496 | 0.981 | 0.958 | 0.500 | 710.9 |

### Moderate

| 샘플러 | MSE_θ | MSE_cal | Spearman | Kendall | top-k | NDCG@k | α̂ | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 13.512 | 1.320 | 0.663 | 0.444 | 0.823 | 0.876 | 0.711 | 16.0 |
| qSGLD | **1.265** | 0.939 | 0.728 | 0.506 | 0.901 | 0.943 | 0.501 | 16.1 |
| cycSGLD | 7.258 | 1.075 | 0.664 | 0.451 | 0.843 | 0.904 | 0.698 | 15.9 |
| SGHMC | 12.392 | 1.551 | 0.607 | 0.401 | 0.783 | 0.852 | 0.701 | 17.0 |
| **AWSGLD** | 1.642 | **0.439** | **0.757** | **0.538** | **0.929** | **0.967** | 0.504 | 16.4 |
| acMH | 1.810 | 1.097 | 0.705 | 0.485 | 0.874 | 0.923 | 0.526 | 686.5 |

### Difficult

| 샘플러 | MSE_θ | MSE_cal | Spearman | Kendall | top-k | NDCG@k | α̂ | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 22.272 | 1.308 | 0.093 | 0.063 | 0.367 | 0.564 | 0.782 | 16.0 |
| qSGLD | 2.227 | 1.153 | 0.327 | 0.220 | 0.501 | 0.716 | 0.667 | 16.6 |
| cycSGLD | 10.904 | 0.938 | −0.556 | −0.364 | 0.061 | 0.337 | 0.769 | 16.9 |
| SGHMC | 20.882 | 1.327 | 0.003 | 0.002 | 0.337 | 0.538 | 0.774 | 20.2 |
| **AWSGLD** | **2.050** | **0.870** | **0.553** | **0.381** | **0.637** | **0.822** | 0.667 | 16.3 |
| acMH | 4.451 | 1.313 | 0.059 | 0.040 | 0.370 | 0.567 | 0.717 | 707.3 |

### Sparse

| 샘플러 | MSE_θ | MSE_cal | Spearman | Kendall | top-k | NDCG@k | α̂ | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 24.647 | 1.476 | −0.064 | −0.042 | 0.271 | 0.500 | 0.761 | 16.6 |
| qSGLD | 1.877 | 1.014 | 0.466 | 0.313 | 0.688 | 0.821 | 0.605 | 16.7 |
| cycSGLD | 12.097 | 0.745 | −0.537 | −0.370 | 0.061 | 0.373 | 0.745 | 16.5 |
| SGHMC | 23.242 | 1.458 | −0.108 | −0.072 | 0.267 | 0.498 | 0.752 | 20.1 |
| **AWSGLD** | **1.731** | **0.629** | **0.606** | **0.407** | **0.863** | **0.918** | 0.598 | 16.9 |
| acMH | 4.624 | 1.433 | 0.125 | 0.084 | 0.411 | 0.615 | 0.687 | 676.8 |

### 부록 — AWSGLD minibatch ablation

`awsgld_minibatch_ablation.py`. Easy 설정, n = 1500, R = 2, T = 5000.

| 설정 | MSE_θ | MSE_cal | Spearman | NDCG@k | top-k | wall(s) |
|---|---|---|---|---|---|---|
| acMH | **2.289** | 1.025 | 0.731 | 0.953 | 0.977 | 3121.6 |
| AWSGLD full-batch | 2.521 | **0.314** | 0.754 | 0.974 | **1.000** | 70.1 |
| AWSGLD batch 750 | 2.549 | 0.316 | **0.762** | **0.975** | **1.000** | 165.3 |
| AWSGLD batch 300 | 2.546 | 0.326 | 0.752 | 0.974 | **1.000** | 67.2 |
| AWSGLD batch 100 | 2.555 | 0.330 | 0.757 | 0.974 | **1.000** | 41.9 |

## 6. 해석

- **AWSGLD가 4개 시나리오 전부에서 Spearman·NDCG 1위**이고, 난이도가 올라갈수록 격차가
  벌어진다. Easy에서는 NDCG 0.969 vs 차선 0.958로 미미하지만 Sparse에서는 0.918 vs 0.821이다.
- **Difficult/Sparse에서 SGLD·cycSGLD·SGHMC는 순위 상관이 0 또는 음수로 붕괴한다.**
  cycSGLD의 −0.556은 단순 실패가 아니라 순위가 뒤집힌 상태로, 저온 탐색 단계에서 잘못된
  basin에 고정된 결과로 보인다.
- **acMH는 Easy/Moderate에서는 경쟁력이 있으나(MSE_θ 최소) Difficult/Sparse에서 Spearman
  0.06~0.12로 무너진다.** 동시에 chain당 ~700초로 Langevin 계열의 40배 이상 비싸다.
- **MSE_θ와 MSE_cal의 괴리에 주의해야 한다.** SGLD는 MSE_θ가 13~24로 크지만 MSE_cal은
  1.3~1.6이다. 즉 절대 스케일이 크게 어긋났을 뿐 Easy에서는 순위가 어느 정도 살아있다.
  반대로 AWSGLD는 두 값이 모두 낮아 스케일과 순위를 함께 맞춘다.
- **minibatch는 사실상 공짜다.** batch 100까지 줄여도 지표 저하가 소수점 셋째 자리 수준이며
  wall time만 70초 → 42초로 줄었다. acMH 대비 74배 빠르면서 NDCG는 0.974 vs 0.953으로 앞선다.
- **α̂ 관점의 부수 관찰** — 난이도가 오르면 모든 샘플러의 α̂가 grid 상한 쪽으로 밀린다.
  Difficult에서 α\* = 0.50인데 SGLD는 0.78을 준다. α 추정은 θ 품질에 종속적이다.

## 7. 재현

```bash
python3 simulation/study_1a/langevin_methods_comparison.py   # 6 sampler × 4 시나리오
python3 simulation/study_1a/merge_acmh_n800.py               # acMH 결과 병합
python3 simulation/study_1a/scenario_landscapes.py           # 시나리오별 energy landscape
python3 simulation/study_1a/awsgld_minibatch_ablation.py     # minibatch ablation (acMH 포함 시 매우 느림)
```

acMH를 포함하면 시나리오·trial당 약 700초가 소요된다. 전체 재실행은 수 시간 단위이다.

## 8. 산출물

| 파일 | 내용 |
|---|---|
| `langevin_methods_comparison_summary.json` | 5 sampler(acMH 제외) 전체 지표 + trial별 값 |
| `langevin_methods_comparison_with_acmh_summary.json` | acMH 병합본 (**본 문서의 근거**) |
| `langevin_methods_comparison_{Easy,Moderate,Difficult,Sparse}.png` | 시나리오별 지표 비교 플롯 |
| `scenario_landscape_{Easy,Moderate,Difficult,Sparse}.png`, `scenario_landscape_all.png` | 사후 에너지 지형 |
| `scenario_landscapes.npz` | θ 격자 + 시나리오별 에너지 값 |
| `awsgld_minibatch_ablation_summary.json`, `_summary.png` | minibatch ablation |
| `_archive/` | 0422/0507 이전 실험 스크립트·플롯·결과 |
