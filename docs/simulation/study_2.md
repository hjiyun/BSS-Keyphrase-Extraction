# Study 2 — 실 키프레이즈 데이터에서의 acMH vs AWSGLD

> 산출물 위치: `simulation/study_2/` (이전 중간 산출물은 `simulation/study_2/archive/`)
> 전처리: `data_JOC/build_baseline.py` → `data_JOC/baseline_preprocessed/`

## 0. 구성

Study 1A~1C는 전부 합성 데이터였다. Study 2는 실 키프레이즈 문서로 넘어가서 세 가지를 묻는다.

| 절 | 질문 | 스크립트 |
|---|---|---|
| **§1 실데이터 벤치마크** | 실 키프레이즈 문서에서 acMH를 이기는가 | `acmh_vs_awsgld_4to10.py`, `dense_*.py`, `hulth_*.py`, `semeval_long5.py` |
| **§2 다봉성 진단** | 실데이터에 자연 다봉이 있기는 한가 | `merge_multimodal.py` |
| **§3 합성 다봉 트랩** | 다봉을 인위 주입하면 우위가 드러나는가 | `trap_multimode(_sharp).py`, `trap_consensus.py` |

결론을 먼저 적으면 **§2가 이 스터디의 분기점**이다. 실 키프레이즈 사후분포에는 자연 다봉이
없고, 따라서 §1에서 AWSGLD의 우위는 mode escape가 아닌 다른 기제에서 나온다.

---

## 1. 실 키프레이즈 데이터 벤치마크

### 1.1 전처리 (`data_JOC/build_baseline.py`)

Wang et al. (2023) §1.5.1.1과 원본 `create_fcm_words`를 재현한다.

```
raw abstract → word_tokenize → POS 태깅 → 명사(NN*) + 형용사(JJ*)만 정점
             → 소문자화 → (옵션) Porter stemming → 후보 단어 시퀀스(순서·중복 보존)
gold(uncontr) → 동일 정규화 → 단어 분해 → 그래프 정점에 있는 것만 truth
그래프 A = window 2 동시출현 (fcm), 대각 0
```

- 출력: `baseline_preprocessed/pre_process/{id}.abstr`, `truth/{id}.uncontr`, `doc_stats.csv`
- 분할: 키워드 ≥ 10 → **dense 391개** (`selected_ids.txt`), < 10 → **sparse 109개** (`sparse_ids.txt`)
- 기본은 stemming 미적용 (논문 Table 1.1과 더 일치)

### 1.2 공통 실험 하네스 (`acmh_vs_awsgld_4to10.py`)

- 그래프: 위 전처리 결과, d = 0.85
- 관측 Y: 정답 중 **k = floor(|truth|/2)** 개를 무작위 관측 (약 50% 은닉)
- u_0 = B⁻¹(1−d)**1**, ini = `base_to_start`(B\*⁻¹Y), α̂ = `alpha_find`(u_0, Y, grid)
- T = 10000, burn-in = 1000, N_SIM = 30
- 두 샘플러는 매 (문서, 시뮬레이션)마다 **동일한 Y / ini / 그래프**를 공유한다
- acMH는 Numba + O(n) 증분 갱신으로 가속했다. `C = ‖B(θ−u_0)‖²`를 성분 제안마다 O(n²)로
  재계산하지 않고 `C_new = C + 2δ(v·B[:,i]) + δ²‖B[:,i]‖²`로 갱신하며, 난수 소비 순서를
  원본과 동일하게 유지해 수학적 동치를 지킨다.
  (프로젝트 README는 약 111배 가속으로 기록하고 있으나, 이를 뒷받침하는 타이밍 산출물이
  저장되어 있지 않다. `--probe` 옵션으로 재측정할 수 있다.)
- 평가: FDR-cutoff γ ∈ {0.05, …, 0.30}별 precision / recall / F / 실현 FDR
  (`force_obs_to_key2`로 Y=1 노드는 π=1로 고정 — 원본과 동일)

### 1.3 정답 4~10개 문서 126편 (`archive/acmh_vs_awsgld_4to10_*`)

126문서 × 30 시뮬레이션 = 3780회.

| γ | 샘플러 | Precision | Recall | F | 실현 FDR | 선택수 |
|---|---|---|---|---|---|---|
| 0.05 | acMH | **0.953** | 0.533 | 0.679 | 0.047 | 4.32 |
| 0.05 | AWSGLD | 0.944 | **0.556** | **0.694** | 0.056 | 4.53 |
| 0.10 | acMH | **0.826** | 0.638 | 0.713 | 0.174 | 5.96 |
| 0.10 | AWSGLD | 0.803 | **0.662** | **0.719** | 0.197 | 6.31 |
| 0.15 | acMH | **0.695** | 0.735 | **0.706** | 0.305 | 8.21 |
| 0.15 | AWSGLD | 0.668 | **0.759** | 0.702 | 0.332 | 8.74 |
| 0.20 | acMH | **0.569** | 0.826 | **0.662** | 0.431 | 11.49 |
| 0.20 | AWSGLD | 0.538 | **0.854** | 0.650 | 0.462 | 12.42 |
| 0.30 | acMH | **0.394** | 0.955 | **0.537** | 0.606 | 21.29 |
| 0.30 | AWSGLD | 0.343 | **0.987** | 0.489 | 0.657 | 25.39 |

**패턴이 명확하다. AWSGLD는 항상 더 많이 선택하고(선택수 ↑) recall을 올리는 대신 precision을
내준다.** F 기준으로는 낮은 γ에서 AWSGLD가, 높은 γ에서 acMH가 이긴다.

### 1.4 dense 문서 (`dense_test.py`, `dense_paper_eval.py`)

dense 문서 5편(1949 / 2007 / 2092 / 215 / 2017), N_SIM = 8, 관측 k = 5 고정, full-batch.
AWSGLD 설정 τ = 1, ζ = 5, DECAY_LR = 100, M_REGIONS = 1000.

**문서별 macro 평균 (`dense_test.csv`)**

| γ | acMH P / R / F | AWSGLD P / R / F |
|---|---|---|
| 0.05 | 0.957 / 0.156 / 0.267 | 0.948 / **0.176** / **0.296** |
| 0.10 | 0.928 / 0.219 / 0.350 | 0.921 / **0.265** / **0.409** |
| 0.15 | 0.859 / 0.297 / 0.434 | **0.890** / **0.381** / **0.531** |
| 0.20 | 0.811 / 0.409 / 0.532 | 0.803 / **0.514** / **0.624** |
| 0.25 | 0.732 / 0.574 / 0.620 | 0.712 / **0.708** / **0.707** |
| 0.30 | 0.662 / 0.753 / 0.678 | 0.614 / **0.956** / **0.742** |

**원논문 방식 pooled(micro) 집계 + ROC/PR (`dense_paper_eval.csv`)**

원논문 `main_graph.R`은 문서별 precision을 평균내지 않고 TP/pos를 문서 합산 후 나눈다.
그 방식으로 다시 집계하고 `precision.recall.auc()`에 해당하는 PR 곡선을 추가했다.

| 지표 | acMH | AWSGLD |
|---|---|---|
| pooled F (γ=0.15) | 0.428 | **0.526** |
| pooled F (γ=0.20) | 0.512 | **0.609** |
| pooled F (γ=0.30) | 0.665 | **0.732** |
| **ROC AUC** | 0.664 | **0.693** |
| PR: recall 0.3에서의 precision | 0.851 | **0.906** |
| PR: recall 0.5에서의 precision | 0.758 | **0.809** |
| PR: recall 0.8에서의 precision | 0.663 | **0.683** |

집계 방식을 원논문식으로 바꿔도 **dense에서는 AWSGLD가 전 구간 우위**이며, γ에 의존하지 않는
ROC AUC와 PR 곡선에서도 이긴다. 즉 dense 우위는 cutoff 선택의 artifact가 아니다.

**minibatch·하이퍼파라미터 ablation** (`dense_minibatch.csv`, `dense_hp_tune.csv`)

- batch를 full → 1/2 → 1/4 → 1/8로 줄여도 γ=0.20 F가 0.5985 / 0.6046 / 0.6041 / 0.6031로
  사실상 불변이다. 실데이터에서도 minibatch는 비용만 줄인다.
- τ(1.0/1.5/2.0), ζ(1/5/10), DECAY_LR(100/200), eps(1.0/2.0) 스윕에서 γ=0.20 F는
  0.5965~0.6155 범위로, **하이퍼파라미터에 둔감하다.** dense 우위가 튜닝의 산물이 아님을 뜻한다.

**6 샘플러 통일 비교 (`dense_allsamplers.py`, N_SIM = 30)**

같은 dense 5문서에 SGLD 계열 4종을 추가해 6 샘플러를 한 표로 비교한다. pooled 집계,
top-k는 문서별 top-20 macro 평균. 실 키프레이즈 사후분포는 **단봉**이므로 Basins/Escape는
정의되지 않고, 순위·추출 지표만 본다.

| 샘플러 | P@20 | R@20 | F@20 | ROC AUC | NDCG@20 |
|---|---|---|---|---|---|
| **AWSGLD** | **0.849** | **0.414** | **0.555** | **0.685** | **0.850** |
| cycSGLD | 0.813 | 0.398 | 0.533 | 0.671 | 0.836 |
| acMH | 0.802 | 0.391 | 0.524 | 0.650 | 0.838 |
| SGLD | 0.759 | 0.370 | 0.496 | 0.626 | 0.803 |
| qSGLD | 0.731 | 0.356 | 0.478 | 0.611 | 0.761 |
| SGHMC | 0.718 | 0.350 | 0.469 | 0.593 | 0.753 |

- **AWSGLD가 P@20·R@20·F@20·ROC AUC·NDCG@20 5개 종합 지표 전부 6 샘플러 중 1위**다.
  합성 트랩(§3)에서 cycSGLD와 대등했던 것과 달리, **자연 단봉 실데이터에서는 AWSGLD가 단독 우위**다.
  데이터·하이퍼파라미터 조작 없이 얻은 결과다.
- 단, γ별 세부에서는 트레이드오프가 있다 (`dense_allsamplers.csv`): **precision·실현 FDR은 cycSGLD·acMH**가
  더 좋고(적게·정확하게 선택), **recall은 AWSGLD가 압도**한다(γ=0.30에서 R 0.949 vs cycSGLD 0.681).
  낮은 γ(0.05~0.10)의 F는 SGLD가 앞선다. 종합 지표(F/AUC/NDCG)에서 AWSGLD가 이기되, "모든 세부
  지표 1위"는 아니라는 점을 병기한다.

### 1.5 dense 랜덤 5문서 심화 (`hulth_*.py`)

고정 시드(20260711)로 dense 풀에서 무작위 추출한 5문서(240 / 275 / 333 / 404 / 2000),
각 10 시드. 관측은 k = floor(|truth|/2).

**기본 비교 (`hulth_rand5_bench.csv`, ALL 행)**

| 샘플러 | Recall | F(γ=0.20) | top-k | ROC AUC |
|---|---|---|---|---|
| acMH | **0.850 ± 0.085** | 0.702 ± 0.057 | **0.719 ± 0.076** | **0.781 ± 0.077** |
| AWSGLD | 0.797 ± 0.077 | **0.705 ± 0.067** | 0.701 ± 0.071 | 0.761 ± 0.084 |

**§1.4의 dense 결과와 방향이 반대다.** 차이는 관측량이다. §1.4는 k = 5 고정이고
여기는 정답의 50%를 관측한다. 관측이 충분하면 AWSGLD의 탐색 이득이 사라지고 acMH가 앞선다.

**ζ 튜닝 (`hulth_zeta_tune.csv`)** — flat-histogram 강도

| 설정 | Recall | F | top-k | AUC |
|---|---|---|---|---|
| acMH | 0.881 | 0.708 | **0.722** | **0.783** |
| AWSGLD ζ=5 (기본) | 0.817 | 0.714 | 0.714 | 0.773 |
| AWSGLD ζ=1 | 0.843 | **0.718** | 0.722 | 0.775 |
| AWSGLD ζ=0 (weighting 끔) | **0.924** | 0.688 | 0.663 | 0.703 |

ζ=0은 recall이 가장 높지만 AUC가 0.703으로 급락한다. adaptive weighting을 끄면 사후분포를
넓게 퍼뜨리기만 하고 순위 정보를 잃는다. **단봉 실데이터에서는 ζ를 약하게(1) 두는 것이 최적**이다.

**eps 스케줄 스윕 (`hulth_eps_tune.csv`)** — ζ=1 고정, ε_k = scale/((t+1)^pow + 10)

| 설정 | Recall | F | top-k | AUC |
|---|---|---|---|---|
| (0.3, 0.6) | 0.843 | **0.718** | **0.722** | **0.775** |
| (0.5, 0.6) | 0.844 | 0.718 | 0.711 | 0.772 |
| (0.8, 0.6) | 0.847 | 0.717 | 0.708 | 0.773 |
| (0.5, 0.5) | 0.847 | 0.717 | 0.715 | 0.775 |

차이가 셋째 자리에 불과해 ε 스케줄은 결정적 요인이 아니다.

**MALA_v2 포함 최대 비교 (`hulth_final_bench.csv`, ALL 행)**

MALA_v2는 MH 보정을 붙여 이산화 편향을 제거한 버전이다.

| 샘플러 | Recall | F | top-k | AUC |
|---|---|---|---|---|
| acMH | 0.850 | 0.702 | **0.719** | **0.781** |
| AWSGLD (ζ=1) | 0.824 | **0.710** | 0.707 | 0.762 |
| MALA_v2 | **0.868** | 0.697 | 0.700 | 0.761 |

**세 샘플러가 사실상 동률**이다 (F 0.697~0.710, AUC 0.761~0.781). 단봉 사후분포에서는
이산화 편향을 없애도(MALA_v2) 이득이 없고, 샘플러 선택 자체가 성능을 좌우하지 않는다.

**약지도 조건 (`hulth_weak.csv`)** — 관측을 정답의 20%로 축소

| 지표 | acMH | AWSGLD ζ=5 | AWSGLD ζ=1 |
|---|---|---|---|
| F (γ=0.20) | 0.559 | **0.623** | 0.617 |
| F (γ=0.25) | 0.607 | **0.653** | 0.651 |
| top-k | 0.599 | 0.633 | **0.636** |
| ROC AUC | 0.639 | 0.674 | **0.676** |

**관측을 줄이자 순위가 다시 뒤집힌다.** 50% 관측(§1.5 기본)에서는 acMH가 이겼는데
20% 관측에서는 AWSGLD가 전 지표 우위다. §1.4(k=5 고정)의 결과와 일관된다.

> **정리** — 실데이터에서 AWSGLD의 우위는 **관측이 희소할수록 커진다.** 라벨 신호가 약해
> 사후분포가 평평해질수록 넓게 탐색하는 성질이 이득이 되고, 라벨이 충분하면 불필요한
> 과탐색이 되어 precision을 깎는다.

### 1.6 SemEval-2010 장문 논문 (`semeval_long5.py`)

논문 §1.5.2 설정을 그대로 재현한다. 스테밍 완료 텍스트 + 빈도 필터(2회 이하 제거),
관측 Y = **제목에서 유래한 키워드**(무작위 아님, 결정적), 정답 = reader 키워드,
그래프 = window 2. 문서 5편(C-42 포함, 나머지 무작위), AWSGLD는 기본 ζ=5(튜닝 없음).

| γ | acMH P / R / F | AWSGLD P / R / F |
|---|---|---|
| 0.05 | **0.950** / 0.251 / 0.389 | 0.530 / **0.336** / **0.408** |
| 0.10 | **0.846** / 0.283 / **0.409** | 0.356 / **0.410** / 0.380 |
| 0.20 | **0.605** / 0.365 / **0.426** | 0.216 / **0.608** / 0.317 |
| 0.30 | **0.389** / 0.476 / **0.386** | 0.121 / **0.815** / 0.209 |
| top-k / AUC | **0.419** / **0.802** | 0.331 / 0.794 |

**AWSGLD가 명확히 진다.** precision이 γ=0.30에서 0.121까지 무너진다. 원인은 두 가지로 보인다.

1. **관측 설계가 다르다.** 제목 유래 관측은 매우 적고 편향돼 있다 (문서당 한 자릿수).
   §1.5의 약지도 결과와 달리, 여기서는 관측이 무작위가 아니라 특정 위치에 몰려 있다.
2. **n이 크다.** 장문 논문은 후보 단어가 수백~수천 개라 AWSGLD의 과탐색이 대량의 FP로 번진다.

다만 ROC AUC는 0.794 vs 0.802로 거의 같다. **순위 자체는 비슷한데 cutoff 위치가 무너진 것**이다.
즉 AWSGLD의 π̂ 분포가 오른쪽으로 밀려 FDR-cutoff가 과선택하는 캘리브레이션 문제다.

---

## 2. 실데이터에 자연 다봉이 있는가 (`merge_multimodal.py`)

**가설** — Hulth 문서들을 어휘를 공유하도록 병합하면 bridge 노드에 상충 인력(frustration)이
생겨 사후분포가 다봉이 된다. 어휘가 안 겹치면 블록 대각 → 분리 가능 → 단봉.

**구성** — 병합 A = Σ_doc (문서별 window-2 인접행렬을 공유 어휘 인덱스로 매핑해 합산),
B = I − d·Gᵀ, u_0 = B⁻¹(1−d)**1**, truth = 문서별 키워드 합집합, Y는 문서마다 자기 키워드의 20%.
α는 `alpha_find(u_0)`로 고정해 U를 θ만의 결정함수로 만든다.

**진단** — ① 다중 재출발 국소최소화(무작위 + 문서 편향 초기값) → 수렴점 군집 → basin 개수,
② 서로 다른 basin 쌍 사이 선형보간 U(θ(λ)) → 장벽 높이.

**조건** — HIGH5(고중첩 5편) / HIGH2(최고중첩 2편) / LOW5(저중첩 5편, 대조군),
에너지는 σ²-explicit(0.5, 1.0)와 σ²-적분 세 가지.

### 결과 (`merge_multimodal.csv`)

| 조건 | 문서수 | n | bridge 노드 | basin 수 | 최대 장벽 |
|---|---|---|---|---|---|
| HIGH5 | 5 | 314 | 70 | **1** | 0.0 |
| HIGH2 | 2 | 113 | 41 | **1** | 0.0 |
| LOW5 | 5 | 178 | 0 | **1** | 0.0 |

세 에너지 정의(σ²=0.5, σ²=1.0, σ²-적분) 전부에서 **basin이 1개**다.

> **가설 기각.** 어휘를 70개나 공유시켜도 사후분포는 단봉이다. BSS 사후분포는
> **단봉 heavy-tail** 구조이며, 문서 병합으로는 자연 다봉이 생기지 않는다.
>
> 이것은 프로젝트 전체 서사를 바꾸는 음성 결과다. **AWSGLD의 mode-escape 강점은 실
> 키프레이즈 데이터에서 자연적으로는 발현될 수 없다.** §1에서 관측된 우위(희소 관측 조건)는
> mode escape가 아니라 탐색 반경이 넓다는 성질 자체의 효과로 해석해야 한다.

---

## 3. 합성 다봉 트랩

§2의 결론에 따라, 다봉 상황에서의 우위를 보이려면 다봉을 **인위 주입**해야 한다.
실그래프(doc 2098, n = 후보 단어 수, 정답 14개)에 K = 5 mode 혼합 사후분포를 얹는다.

```
U_k(θ)   = −loglik(θ) + ‖B(θ − u^(k))‖² / (2σ²),  σ² = 2.0
U_mix(θ) = −log Σ_k exp(−U_k(θ))            ← log-sum-exp (CLAUDE.md 준수)
```

T = 50000, burn = 5000. 평가는 top-k와 ROC AUC를 주로 본다.
FDR-cutoff는 `force_obs_to_key2`가 Y=1을 π=1로 고정하는 탓에 precision=1.0 인공물이 생겨
트랩 실험에서는 신뢰도가 낮다.

### 3.1 5모드 분산 트랩 (`trap_multimode.py`)

정답 키워드 14개를 5개 모드에 라운드로빈으로 흩뿌린다(모드당 ~3개). 한 모드에 갇히면
그 모드의 3개밖에 못 잡는 구조다.

| k | acMH P / R / F | AWSGLD P / R / F |
|---|---|---|
| 5 | **0.733** / 0.262 / **0.386** | 0.600 / 0.214 / 0.316 |
| 10 | **0.633** / 0.452 / **0.528** | 0.533 / 0.381 / 0.444 |
| 16 | 0.479 / 0.548 / 0.511 | 0.479 / 0.548 / 0.511 |
| 20 | 0.383 / 0.548 / 0.451 | **0.400** / **0.571** / **0.471** |
| **ROC AUC** | 0.563 | **0.612** |

**의도한 결과가 나오지 않았다.** top-k에서는 acMH가 이기고 AWSGLD는 AUC에서만 앞선다.
`trap_multimode_sharp.py`(저온 + 강한 flat-histogram)로 탈출을 강화해도 AWSGLD의 top-k는
오히려 더 나빠진다 (k=10에서 P 0.533 → 0.467).

원인은 **커버리지와 선명도의 트레이드오프**다. 여러 모드를 방문하려면 온도를 올려야 하는데,
그 온도에서는 π̂가 평탄해져 상위권 변별력이 사라진다. 탈출에 성공한 지점이 곧 순위가
흐려지는 지점이다.

### 3.2 consensus 트랩 (`trap_consensus.py`)

트랩 위상을 바꾼다.

- **진짜 키워드 14개 = 모든 모드에서 공통으로 높음** (COMMON = 2.0)
- **각 모드마다 고유한 미끼 비키워드를 더 높게 배치** (DECOY = 4.0)

한 모드에 갇히면 그 모드의 미끼에 오염되고, 여러 모드를 평균내면 미끼는 상쇄되고
공통 키워드만 살아남는 구조다.

| k | acMH P / R / F | AWSGLD P / R / F |
|---|---|---|
| 5 | **0.000** / 0.000 / 0.000 | **0.600** / **0.214** / **0.316** |
| 8 | 0.250 / 0.143 / 0.182 | **0.500** / **0.286** / **0.364** |
| 10 | 0.400 / 0.286 / 0.333 | **0.533** / **0.381** / **0.444** |
| 12 | 0.500 / 0.429 / 0.462 | **0.583** / **0.500** / **0.538** |
| 14 | **0.571** / **0.571** / **0.571** | 0.548 / 0.548 / 0.548 |
| 20 | **0.483** / **0.690** / **0.569** | 0.450 / 0.643 / 0.529 |
| **ROC AUC** | 0.571 | **0.668** |

**acMH의 top-5 precision이 0.000이다.** 갇힌 모드의 미끼 4개가 상위권을 완전히 점령한다.
AWSGLD는 0.600으로 상위 5개 중 3개가 진짜 키워드다. k ≥ 14에서 역전되는 것은 AWSGLD가
넓게 퍼진 만큼 하위권이 흐려지기 때문이다.

FDR-cutoff에서는 대비가 극단적이다 (`trap_consensus_fdr.csv`). acMH는 전 γ에서
precision 0.000 / 실현 FDR 1.000, 즉 **선택한 것이 전부 미끼**다. AWSGLD는 γ=0.05에서
precision 0.688을 낸다.

`trap_consensus_words.py`로 실제 선택 단어를 출력해 이 해석을 육안 확인할 수 있다.

> **AWSGLD의 mode-escape 우위는 "모드마다 정답이 흩어진" 위상이 아니라 "정답이 모드 공통
> 신호이고 오답이 모드 고유" 위상에서 드러난다.** 전자는 탈출해도 온도 대가로 상쇄되지만,
> 후자는 여러 모드를 평균내는 것 자체가 잡음 제거로 작동한다.

---

## 4. 종합 해석

1. **실 키프레이즈 사후분포는 단봉 heavy-tail이다** (§2). 문서를 병합해 어휘를 70개 공유시켜도
   basin은 1개다. 다봉을 전제한 방법론 서사는 실데이터만으로는 성립하지 않는다.
2. **그럼에도 dense·희소관측 조건에서 AWSGLD가 이긴다** (§1.4, §1.5 약지도). γ에 무관한
   ROC AUC(0.693 vs 0.664)와 PR 곡선에서도 이기므로 cutoff artifact가 아니다.
   기제는 mode escape가 아니라 **넓은 탐색 반경 → recall 확보**이다.
3. **관측이 충분하면 그 성질이 부담이 된다** (§1.5 기본 50% 관측, §1.6 SemEval).
   특히 SemEval 장문에서는 precision이 0.12까지 붕괴한다. 순위(AUC)는 비슷한데 cutoff가
   무너지는 **캘리브레이션 문제**다.
4. **다봉을 주입하면 위상에 따라 결과가 갈린다** (§3). 정답 분산형에서는 커버리지-선명도
   트레이드오프로 이득이 없고, consensus형에서는 acMH가 top-5 precision 0.000으로
   완전히 실패하는 반면 AWSGLD는 0.600을 낸다.

이 지점에서 Study 3이 출발한다. **§3의 트랩은 중심(mode center)을 인위로 지어냈다.**
Study 3은 중심을 실데이터에서 산출된 값(문서별 TextRank 해)으로만 구성해 같은 질문을 다시 던진다.

---

## 5. 재현

```bash
# 전처리 (원논문 POS 필터 재현 + dense/sparse 분할)
python3 data_JOC/reproduce_pos_filter.py
python3 data_JOC/build_baseline.py

# §1 실데이터 벤치마크
python3 simulation/study_2/acmh_vs_awsgld_4to10.py       # 126문서 (--probe 로 타이밍만)
python3 simulation/study_2/dense_test.py
python3 simulation/study_2/dense_paper_eval.py           # 원논문 pooled 집계 + ROC/PR
python3 simulation/study_2/dense_allsamplers.py          # dense 6 샘플러 통일 비교 (~48분)
python3 simulation/study_2/dense_minibatch.py
python3 simulation/study_2/dense_hp_tune.py
python3 simulation/study_2/hulth_rand5_bench.py
python3 simulation/study_2/hulth_zeta_tune.py
python3 simulation/study_2/hulth_eps_tune.py
python3 simulation/study_2/hulth_final_bench.py          # + MALA_v2
python3 simulation/study_2/hulth_bygamma.py
python3 simulation/study_2/hulth_weak.py                 # 약지도 20% 관측
python3 data_JOC/extract_semeval_titles.py               # 제목 추출 (선행)
python3 simulation/study_2/semeval_long5.py

# §2 다봉성 진단
python3 simulation/study_2/merge_multimodal.py

# §3 합성 다봉 트랩
python3 simulation/study_2/trap_multimode.py
python3 simulation/study_2/trap_multimode_sharp.py
python3 simulation/study_2/trap_consensus.py
python3 simulation/study_2/trap_consensus_words.py       # 선택 단어 확인
```

## 6. 산출물

| 파일 | 내용 |
|---|---|
| `archive/acmh_vs_awsgld_4to10_{summary.json,by_gamma.csv,results.jsonl}` | 126문서 벤치마크 |
| `dense_test.csv`, `dense_paper_eval.csv` | dense 5문서 macro / pooled 평가 |
| `dense_allsamplers.py`, `dense_allsamplers_summary.csv`, `dense_allsamplers.csv` | dense 6 샘플러 통일 비교 (P/R/F@20 + AUC + NDCG@20, γ별 P/R/F/FDR) |
| `dense_minibatch.csv`, `dense_hp_tune.csv` | batch·하이퍼파라미터 ablation |
| `hulth_rand5_bench.csv`, `hulth_zeta_tune.csv`, `hulth_eps_tune.csv`, `hulth_final_bench.csv`, `hulth_bygamma.csv`, `hulth_weak.csv` | dense 랜덤 5문서 심화 |
| `semeval_long5.csv` | SemEval-2010 장문 5편 |
| `merge_multimodal.csv` | **병합 다봉성 진단 (음성 결과)** |
| `trap_multimode_{topk,fdr,auc}.csv`, `trap_multimode_sharp_*.csv` | 5모드 분산 트랩 |
| `trap_consensus_{topk,fdr,auc}.csv` | consensus 트랩 |
| `*_result.png`, `dense_landscape_1d.png` | 트랩·지형 시각화 |
| `archive/` | sparse10 계열, MALA 비교, floor/gradclip/precond 스캔 등 이전 중간 산출물. **위 126문서 근거 파일 2개만 git 추적**이고 나머지는 로컬 보관이다 (`.gitignore` 참고) |

> **범위 밖** — `awsgld_convergence.py`와 `awsgld_convergence_{easy,moderate,difficult}.png`
> (AWSGLD vs SGHMC 4-panel 수렴 진단)는 본 문서에서 제외했다. 파일은 `simulation/study_2/`에
> 그대로 남아 있다.
