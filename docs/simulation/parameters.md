# 시뮬레이션 폴더별 실험·파라미터 정리

`simulation/` 하위 각 폴더의 스크립트, 목적, 핵심 파라미터, 산출물을 한 표로 정리한다.
서술형 해석은 각 스터디 문서([study_1a](study_1a.md) … [study_3](study_3.md))에 있고,
이 문서는 **"무엇을 어떤 값으로 돌렸나"의 참조표**이다.

공통 표기: T = chain 길이, burn = burn-in, R/seed = 반복 수, d = damping,
α = PU 라벨잡음, ζ = AWSGLD flat-histogram 강도, σ²_floor = σ² 하한.

---

## study_1a — 난이도 sweep (합성 n=800)

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `langevin_methods_comparison.py` | 6 sampler × 4 시나리오 θ 복원 비교 | T=5000, burn=1000, R=3, d=0.85, batch=100, seed_base=20260507 | `langevin_methods_comparison_with_acmh_summary.json`, `*_{Easy,Moderate,Difficult,Sparse}.png` |
| `scenario_landscapes.py` | 시나리오별 posterior energy 지형 시각화 | 위 SCENARIOS 와 동일 μ/σ/α | `scenario_landscape_*.png`, `scenario_landscapes.npz` |
| `awsgld_minibatch_ablation.py` | AWSGLD batch-size ablation | n=1500, T=5000, R=2, batch ∈ {full,750,300,100} | `awsgld_minibatch_ablation_summary.json` |
| `merge_acmh_n800.py` | acMH 결과를 5-sampler JSON 에 병합 | — | (with_acmh_summary.json 생성) |

**시나리오 파라미터** (n=800 공통, ρ = S/W/N 비율)

| 시나리오 | ρ | μ_S/μ_W/μ_N | σ_θ | α | block (within/sw/other) |
|---|---|---|---|---|---|
| Easy | 0.20/0.20/0.60 | 2.5/1.0/−2.5 | 0.35 | 0.20 | 0.20/0.03/0.005 |
| Moderate | 0.20/0.20/0.60 | 2.0/0.5/−1.8 | 0.50 | 0.35 | 0.20/0.03/0.005 |
| Difficult | 0.20/0.20/0.60 | 1.5/0.0/−1.0 | 0.60 | 0.50 | 0.15/0.05/0.01 |
| Sparse | 0.10/0.18/0.72 | 2.0/1.0/−1.0 | 0.55 | 0.40 | 0.20/0.03/0.005 |

**샘플러 하이퍼파라미터** (전 스터디 공통 기본값)

| 샘플러 | 학습률 / 스텝 | 기타 |
|---|---|---|
| SGLD | ε_k = 0.02/((t+1)^0.6+10) | τ=1 |
| qSGLD | ε_k = 0.3/((t+1)^0.6+10) | P≈(BᵀB)⁻¹, Cholesky 잡음 |
| cycSGLD | ε_base = 0.01 | 10 cycle, 저온 τ/1e4 |
| SGHMC | η_base = 0.01 | friction=0.1, τ=1, M=I, B̂=0 |
| AWSGLD | ε_k = 0.3/((t+1)^0.6+10) | τ=1, ζ=5, M_REGIONS=1000, σ²_floor=0.5 |
| acMH | proposal cov ∝ (BᵀB)⁻¹σ²·4/n | 원본 `keyphrase_functions.gibbs_mh` |

---

## study_1b — local trap escape (합성 n=400)

타깃 분포는 `local_trap_landscape.py`의 `PARAMS`가 단일 진실 원천이다.

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `local_trap_landscape.py` | 3-mode mixture posterior 정의·시각화 + 장벽 높이 | PARAMS (아래) | `local_trap_landscape.{png,npz}` |
| `data_generator.py` | seed별 데이터 생성 (SBM + label conflict) | n=400, p_in=0.40, p_out=0.005, d=0.90, flip S→0=0.30, N→1=0.10 | `data_seed{0..4}.npz` |
| `data_landscape_overview.py` | θ\*/Y 분포 개요 시각화 | — | `data_landscape_overview.png` |
| `acmh_vs_awsgld.py` | acMH vs AWSGLD 복원·탈출 | T=5000, burn=500, 3 chain, batch=100, **σ²_floor=1.0**, bad init θ⁰=μ_N | `ava_results.npz`, `ava_metric_summary.json` |
| `sgld_only.py` | SGLD/qSGLD/cycSGLD/SGHMC | T=5000, burn=500, 3 chain, batch=100, σ²_floor=0.5 | `sgld_results.npz`, `sgld_metric_summary.json` |

**PARAMS** (타깃 분포): μ_S/μ_W/μ_N = 2.5/1.0/**−0.8**, σ_θ=0.26, α=0.20, ρ=0.20/0.20/0.60

- Study 1A 대비: damping 0.85→**0.90**(prior 강화), label conflict 48/400 주입, μ_N 을 얕게
- θ̂ = chain 0 post-burn 평균

> `_archive/`: σ² sweep, 다봉성 검증(GD), NDCG/ESS/혼동행렬 보조, seed 1~4 데이터

---

## study_1c — 규모 × 다중 seed (합성, acMH 제외)

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `data_generator.py` | 규모·seed별 데이터 생성 | n ∈ {200,1500,10000} × seed 0–4, μ_N=**−1.5**, d=0.85, flip 0.10/0.05 | `data_n{n}_seed{s}.npz` |
| `data_landscape_overview.py` | 분포 개요 | — | `data_landscape_overview.png` |
| `sampler_comparison.py` | 5 sampler 비교 (3 chain pooled) | 아래 SCALE_CFG, σ²_floor=1.0, burn=T×10% | `results_n{n}_multiseed_summary.json` |
| `plot_n_scaling.py`, `plot_aw_gap_vs_n.py`, `plot_aw_diff_n200.py` | 규모별 성능 추이 시각화 | — | `n_scaling_comparison.png` 등 |

**SCALE_CFG** (규모별)

| n | T | batch | NDCG@k | σ_θ | p_in | p_out |
|---|---|---|---|---|---|---|
| 200 | 5000 | 50 | 40 | 0.20 | 0.50 | 0.003 |
| 1500 | 10000 | 200 | 150 | 0.26 | 0.10 | 0.001 |
| 10000 | 10000 | 1000 | 1000 | 0.30 | 0.015 | 0.0002 |

- θ̂ = **3 chain pooled mean** (Study 1B의 single-chain lucky 문제 제거 목적)
- bad init θ⁰=μ_N + dispersed(μ_W, μ_S)
- n=10000 은 seed 0 단일 (chain당 ~3200–3600초)

---

## study_2 — 실 키프레이즈 데이터

전처리: `data_JOC/build_baseline.py` → dense(≥10) 391편 / sparse(<10) 109편. d=0.85, window=2.
공통 하네스: `acmh_vs_awsgld_4to10.py` (T=10000, burn=1000, N_SIM=30, FDR γ∈{0.05…0.30}).
acMH는 Numba + O(n) 증분 갱신으로 가속.

### 실데이터 벤치마크

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `acmh_vs_awsgld_4to10.py` | 정답 4~10개 126문서 벤치마크 | 관측 k=floor(truth/2)≈50%, N_SIM=30 | `archive/acmh_vs_awsgld_4to10_{summary.json,by_gamma.csv}` |
| `dense_test.py` | dense 5문서 macro 평가 + 분포 진단 | 문서 {1949,2007,2092,215,2017}, N_SIM=8, k=5, ζ=5 | `dense_test.csv` |
| `dense_paper_eval.py` | 원논문 pooled(micro) 집계 + ROC/PR | 동일 5문서, τ=1, ζ=5, DECAY_LR=100 | `dense_paper_eval.csv` |
| `dense_minibatch.py` | batch ablation (full/½/¼/⅛) | dense 5문서 | `dense_minibatch.csv` |
| `dense_hp_tune.py` | τ/ζ/DECAY/eps 스윕 | τ∈{1,1.5,2}, ζ∈{1,5,10}, eps∈{1,2} | `dense_hp_tune.csv` |
| `hulth_rand5_bench.py` | dense 랜덤 5문서 기본 비교 | 문서 {240,275,333,404,2000}(seed 20260711), 10 seed, k≈50% | `hulth_rand5_bench.csv` |
| `hulth_zeta_tune.py` | ζ 스윕 (flat-histogram 강도) | ζ ∈ {5,1,0}, 5 seed | `hulth_zeta_tune.csv` |
| `hulth_eps_tune.py` | ε 스케줄 스윕 (ζ=1 고정) | (scale,pow) ∈ {(.3,.6),(.5,.6),(.8,.6),(.5,.5)} | `hulth_eps_tune.csv` |
| `hulth_final_bench.py` | acMH vs AWSGLD(ζ=1) vs MALA_v2 | 10 seed | `hulth_final_bench.csv` |
| `hulth_bygamma.py` | γ별 FDR-cutoff 상세 | ζ=1, 10 seed, γ∈{.05….30} | `hulth_bygamma.csv` |
| `hulth_weak.py` | 약지도 (관측 20%) | RATIO=0.20, ζ∈{5,1}, 10 seed | `hulth_weak.csv` |
| `semeval_long5.py` | SemEval-2010 장문 5편 | 관측 = 제목 유래(결정적), ζ=5, MINCOUNT=3, 10 seed | `semeval_long5.csv` |

### 다봉성 진단 / 합성 트랩 / 수렴 진단

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `merge_multimodal.py` | 문서 병합의 자연 다봉성 검증 (**음성 결과**) | HIGH5/HIGH2/LOW5, σ²∈{0.5,1.0,적분}, 관측 20% | `merge_multimodal.csv` |
| `trap_multimode.py` | 5모드 분산 트랩 (정답을 모드마다 흩뿌림) | doc 2098, K=5, σ²=2.0, T=50000, burn=5000, SEP 자동탐색, TAU 자동탐색 | `trap_multimode_{topk,fdr,auc}.csv` |
| `trap_multimode_sharp.py` | 저온 + 강 flat-histogram 변형 | 위 + 강화 설정 | `trap_multimode_sharp_*.csv` |
| `trap_consensus.py` | consensus 트랩 (정답 공통, 미끼 모드고유) | K=5, σ²=2.0, COMMON=2.0, DECOY=4.0, T=50000 | `trap_consensus_{topk,fdr,auc}.csv` |
| `trap_consensus_words.py` | consensus 트랩 선택 단어 출력 | 위와 동일 | (stdout) |
| `awsgld_convergence.py` | AWSGLD vs SGHMC 4-panel 수렴 진단 (**보고서 범위 밖**) | Easy/Moderate/Difficult | `awsgld_convergence_*.png` |

> `archive/`: sparse10 계열, MALA 비교, floor/gradclip/precond 스캔 등 폐기된 중간 산출물.
> 126문서 근거 파일 2개만 git 추적.

---

## study_3 — 실데이터로 구성한 local trap

중심 u^(k) = 문서별 TextRank 해. `U_mix = −logsumexp(−U_k)`.
두 샘플러가 같은 타깃·온도·(BᵀB)⁻¹ 기하를 쓰므로 차이는 flat-histogram 유무뿐.
**최종 설정: BASELINE=−2.0, σ²=0.5** (초기 스캔 −0.5 는 basin 뭉개짐 → 실패).

| 스크립트 | 목적 | 핵심 파라미터 | 산출물 |
|---|---|---|---|
| `trap_build.py` | 트랩 구성 + 다봉성 검증 스캔 | K∈{3,5}, σ²∈{1,2,4}, d=0.85, 관측 20%, 어휘 중첩 낮은 조합 우선 | `trap_build.csv` |
| `trap_landscape.py` | 3문서 에너지 지형 (1D 단면) | 문서 {1994,212,227}, n=118, basin 3, 장벽 29.9/7.9 | `trap_landscape.{png,npz}` |
| `trap10_landscape.py` | 10문서 에너지 지형 | n=322, basin 10, 장벽 3.3~22.5 | `trap10_landscape.{png,npz}` |
| `trap_samplers.py` | acMH / AWSGLD 공정 비교 구현 | acMH: θ\*~N(θ,s²P); AWSGLD: P∇U + √(2τε)L noise | (모듈) |
| `tune.py` | 3문서 하이퍼파라미터 스캔 | step/TAU/eps0/ZETA 스윕 | (stdout), `tuned.npz` |
| `recall_tune.py` | recall 지향 재튜닝 | TAU×ZETA×T, eps0=12 고정 | (stdout) |
| `tune10.py` | 10문서 재튜닝 | step/eps0×ζ 스윕 | `tuned10.npz` |
| `run_experiment.py` | 3문서 탐색·탈출 | 아래 3문서 설정, 10 seed | `experiment_result.npz` |
| `run_metrics.py` | 3문서 전 지표 (표만 출력) | 동일 | `run_metrics.log` |
| `run10_metrics.py` | 10문서 전 지표 | 아래 10문서 설정, 10 seed | `result10.npz`, `run10_metrics.log` |
| `run_trace.py` | 모드 방문 트레이스 시각화 | 대표 seed | `mode_visit.png` |

**실행 설정**

| 항목 | 3문서 | 10문서 |
|---|---|---|
| 문서 | 1994, 212, 227 | 1994,212,227,233,260,1982,2117,206,374,403 |
| n / 정답 / 관측 Y | 118 / 49 / 9 | 322 / 156 / 30 |
| τ | 1.0 | 1.0 |
| acMH step | 0.3 | 0.1 (0.3↑은 수용률 0) |
| AWSGLD ε₀ / ζ | 12.0 / 10.0 | 12.0 / 20.0 |
| T / burn | 12000 / 3000 | 20000 / 5000 |
| 출발 basin | doc 212 (가운데) | doc 1982 (가운데) |
| seed | 10 | 10 |

> `run_metrics.py` docstring 은 ζ=5 로 적혀 있으나 실제 코드 상수는 **ζ=10**.
> `tuned.npz` 는 초기 스캔 결과이며 최종 설정 아님.

---

## 폴더 요약 한 줄

| 폴더 | 데이터 | 한 줄 |
|---|---|---|
| study_1a | 합성 n=800 | 난이도 4단계 × 6 sampler — 누가 먼저 무너지는가 |
| study_1b | 합성 n=400 | bad init 에서 trap 탈출을 직접 측정 |
| study_1c | 합성 n=200~10000 | 규모·seed 견고성 (3 chain pooled) |
| study_2 | Hulth·SemEval | 실데이터 벤치마크 + 다봉성 진단(음성) + 합성 트랩 |
| study_3 | Hulth | 중심을 지어내지 않은 실데이터 local trap |
