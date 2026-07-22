# 시뮬레이션 보고서 — BSS + AWSGLD

Wang et al. (2023)의 Bayesian Semi-supervised (BSS) 키프레이즈 추출에서 사후추론 샘플러를
**AWSGLD** (Adaptively Weighted SGLD; Liang et al. 2022)로 대체했을 때의 효과를 검증한
시뮬레이션 전체 기록이다. 각 스터디의 설계·설정·결과·해석을 개별 문서로 정리했다.

문서는 **디스크에 저장된 산출물(JSON / CSV / npz / log)을 근거로만** 작성했다.
근거 파일이 없던 항목은 재실행해 채웠고, 그 사실을 각 문서에 명시했다.

## 문서

| 문서 | 다루는 것 | 데이터 |
|---|---|---|
| [Study 1A](study_1a.md) | 난이도 4단계 × 6 샘플러 — 어떤 샘플러가 먼저 무너지는가 | 합성 (n=800) |
| [Study 1B](study_1b.md) | 다봉 사후분포에서 local trap 탈출 — 원인을 하나로 좁힘 | 합성 (n=400) |
| [Study 1C](study_1c.md) | 규모(n=200/1500/10000) × 다중 seed — 결론의 견고성 | 합성 |
| [Study 2](study_2.md) | 실 키프레이즈 데이터 벤치마크 + 다봉성 진단 + 합성 트랩 | Hulth, SemEval |
| [Study 3](study_3.md) | 실데이터로 구성한 local trap에서의 acMH vs AWSGLD | Hulth |

## 이야기의 흐름

각 스터디는 앞 스터디가 남긴 질문에서 출발한다.

```
1A  난이도를 올리면 SGLD·cycSGLD·SGHMC·acMH가 무너지고 AWSGLD만 버틴다
      → 왜 버티는가?
1B  bad init에서 trap 탈출을 직접 측정. preconditioning이 1차 요인,
    adaptive weighting이 2차(탈출 이후의 혼합 품질) 요인
      → 이 결론이 규모와 seed에 견고한가?
1C  n=200~10000, seed 0~4에서 재현. 다만 θ̂ 정의를 pooled mean으로 바꾸자
    1B의 cycSGLD 이상 결과가 artifact였음이 드러남
      → 실데이터에서도 성립하는가?
2   실 키프레이즈 사후분포는 단봉 heavy-tail이며 자연 다봉이 없다(음성 결과).
    그럼에도 관측이 희소하면 AWSGLD가 이긴다. 다봉을 인위 주입하면
    위상(consensus형)에 따라 우위가 드러난다
      → 중심을 지어내지 않고도 같은 결론이 나오는가?
3   mode 중심을 문서별 TextRank 해로 두어 실데이터 지형 구성.
    acMH는 탈출율 0.00, AWSGLD는 0.80. P@20 0.555 → 0.745
```

## 비교 샘플러

| 샘플러 | 설명 | 등장 |
|---|---|---|
| **acMH-within-Gibbs** | Wang et al. (2023) BSS 원본. (BᵀB)⁻¹σ² proposal + MH 수락 | 1A, 1B, 2, 3 |
| **SGLD** | Welling & Teh (2011) vanilla | 1A, 1B, 1C |
| **qSGLD** | (BᵀB)⁻¹ preconditioning + Cholesky 상관 잡음 | 1A, 1B, 1C |
| **cycSGLD** | Zhang et al. (2020) 순환 학습률 + 2단계 온도 | 1A, 1B, 1C |
| **SGHMC** | Chen et al. (2014) 보조 운동량 + 마찰 | 1A, 1B, 1C, 2 |
| **MALA v1/v2** | MH 보정으로 이산화 편향 제거 | 2 |
| **AWSGLD** | Liang et al. (2022) 에너지 분할 위 adaptive weighting + preconditioning. **본 연구 제안** | 전부 |

## 지표 정의

| 지표 | 정의 | 주의점 |
|---|---|---|
| **MSE_θ** | mean((θ̂ − θ\*)²), logit 공간 | 스케일 왜곡에 민감 |
| **MSE_cal** | θ̂를 θ\*에 1차 회귀 보정 후의 MSE | 순위는 맞고 스케일만 틀린 경우를 분리 |
| **Spearman / Kendall** | θ̂ vs θ\* 순위 상관 | 음수면 순위가 뒤집힌 것 |
| **NDCG@k** | θ\* 순위를 [0,1]로 정규화한 graded relevance의 상위 k 품질 | k가 스터디마다 다름 |
| **top-k P/R/F** | 상위 k개와 정답 집합의 precision / recall / F | γ에 무관 |
| **ROC AUC** | π̂ 순위 기반 | **cutoff 선택과 무관해 트랩 실험에서 가장 신뢰도가 높음** |
| **FDR-cutoff P/R/F** | 명목 FDR γ에서 자른 뒤의 성능 + 실현 FDR | `force_obs_to_key2`가 Y=1을 π=1로 고정 → 트랩 실험에서 precision=1.0 인공물 발생 |
| **R̂** | Gelman-Rubin, 노드별 median / q90 / max | < 1.1 양호, > 1.2 수렴 실패 |
| **ESS** | Geyer initial positive sequence, 노드별 중앙값 | |
| **Cost/ESS** | wall time ÷ ESS median | 시간당 정보 비용 |

## 핵심 결과 요약

| 스터디 | 한 줄 결론 | 대표 수치 |
|---|---|---|
| 1A | 난이도가 올라갈수록 AWSGLD 격차 확대 | Sparse NDCG 0.918 vs 차선 0.821 |
| 1B | trap 탈출은 preconditioning이 1차, adaptive weighting이 2차 | S basin 도달 AWSGLD 80/80(17 step) vs acMH 0/80 |
| 1C | 전 규모에서 MSE·R̂·ESS 1위, 규모에 견고 | MSE 1.005 / 1.232 / 1.251 (n=200/1500/10000) |
| 2 | **실데이터는 단봉** — 우위는 희소 관측에서만 나옴 | 병합 다봉성 진단 basin=1 (전 조건) |
| 2 | dense·희소관측에서는 γ 무관하게 우위 | ROC AUC 0.693 vs 0.664 (pooled) |
| 2 | consensus형 트랩에서 acMH 완전 실패 | top-5 precision 0.000 vs 0.600 |
| 3 | 실데이터 지형에서 acMH는 탈출 불가 | 탈출율 0.00 vs 0.80, P@20 0.555 → 0.745 |

## 설계 원칙 (`CLAUDE.md` 금지 사항)

전 스터디가 다음을 지킨다.

1. **Oracle 초기화 금지** — u_0 = θ\*를 쓰지 않는다. 항상 `u_0 = B⁻¹(1−d)1`.
   쓰면 시나리오 난이도 차이가 마스킹된다.
2. **A-noise로 샘플러 비교 금지** — 두 샘플러가 동일한 그래프·라벨·초기값을 받으므로
   그래프 잡음은 비교에 아무 정보를 주지 않는다.
3. **사후 에너지를 직접 가중합으로 계산 금지** — 항상 log-sum-exp mixture를 쓴다.
   가중합으로 만들면 항상 단일 minimum이 생겨 다봉 실험이 성립하지 않는다.

## 문서화 과정에서 확인한 불일치

보고서 작성 시 참고할 사항이다. 결과 자체를 뒤집는 것은 없다.

| 위치 | 내용 |
|---|---|
| `study_1b/ava_metric_summary.json` | `bad_init` 설명 문자열이 `mu_N = -1.0`으로 남아 있음. 같은 파일 `mu_map`은 `-0.8`이고 이쪽이 실제값 |
| `study_1b/local_trap_landscape.py` | docstring은 μ_N = −1.5로 설명하나 `PARAMS`의 실제값은 −0.8, σ_θ는 0.26 |
| `study_1b/ndcg_summary.json` | SGHMC 누락 (`_archive/ndcg_at_k.py`의 대상 목록에 없음). 본 문서는 저장된 체인에서 재계산 |
| `study_1b/_archive/extra_metrics_summary.json` | Cost/ESS가 이전 실행의 wall time으로 계산됨. 본 문서는 현재 wall time으로 재계산 (qSGLD 0.21 → 0.333, cycSGLD 0.51 → 0.841) |
| `study_3/run_metrics.py` | docstring은 ζ=5, 실제 코드 상수는 ζ=10 |
| `study_3/tuned.npz` | `tune.py`의 초기 스캔 결과이며 최종 설정이 아님 |
| `study_3/run_metrics.py`, `run10_metrics.py` | 지표 표를 파일로 저장하지 않음 → 재실행해 `*.log`로 갈무리 |

## 관련 위치

| 경로 | 내용 |
|---|---|
| `code_JOC/keyphrase_functions_awsgld.py` | AWSGLD `gibbs_mh` (σ²_floor 매개변수화) |
| `code_JOC/original/keyphrase_functions.py` | Wang et al. (2023) 원본 acMH (수정 금지) |
| `code_JOC/mala_keyphrase.py` | MALA v1/v2 |
| `code_JOC/awsgld_tunable.py`, `awsgld_coordvar.py` | ε 스케줄 파라미터화 / 단어별 분산 preconditioner |
| `data_JOC/build_baseline.py` | 원논문 POS 전처리 재현 + dense/sparse 분할 |
| `data_JOC/baseline_preprocessed/` | 전처리 결과 (dense 391 / sparse 109) |
| `simulation/BSS_032926.tex` | LaTeX 원고 (Study 1A/1B/1C/2 섹션 — **Study 3 미반영**) |

## 남은 과제

- [ ] SGHMC 하이퍼파라미터(학습률 / friction) 튜닝 — 현재는 Study 1A 기본값 고정
- [ ] Study 3 트랩의 `BASELINE` / `σ²` 민감도 분석 — 다봉 형성을 위한 선택값이라 인위성이 남아 있음
- [ ] Study 3 결과를 `BSS_032926.tex`에 반영
- [ ] SemEval 장문에서 AWSGLD의 FDR-cutoff 캘리브레이션 붕괴 원인 규명 (AUC는 대등한데 cutoff만 무너짐)
- [ ] n = 10000 다중 seed 확장

## 참고 문헌

- Wang et al. (2023), *Bayesian Semi-supervised Keyphrase Extraction*, INFORMS Journal on Computing.
- Liang et al. (2022), *Adaptively Weighted Stochastic Gradient MCMC*.
- Chen et al. (2014), *Stochastic Gradient Hamiltonian Monte Carlo*, ICML.
- Welling & Teh (2011), *Bayesian Learning via Stochastic Gradient Langevin Dynamics*, ICML.
- Zhang et al. (2020), *Cyclical Stochastic Gradient MCMC for Bayesian Deep Learning*, ICLR.
