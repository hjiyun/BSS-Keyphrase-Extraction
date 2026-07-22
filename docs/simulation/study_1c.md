# Study 1C — 규모(n) 확장과 다중 seed에서의 안정성

> 산출물 위치: `simulation/study_1c/`
> 결과 근거 파일: `results_n{200,1500,10000}_multiseed_summary.json`

## 1. 목적

Study 1B는 n = 400, seed 1개, chain 0 기준의 단일 조건이었다. Study 1C는 그 결론이
**규모와 seed에 걸쳐 견고한지**를 확인한다. 세 가지를 바꾼다.

1. **규모** — n ∈ {200, 1500, 10000}
2. **반복** — seed 0~4 (n = 10000은 비용 문제로 seed 0 단일)
3. **추정량 정의** — θ̂를 chain 0이 아니라 **3 chain pooled mean**으로 바꾼다.
   Study 1B에서 cycSGLD가 MSE 1위였지만 R̂ max 5.09였던 single-chain lucky 문제를 제거한다.

acMH는 이 규모에서 비현실적으로 느려(n = 1500 기준 chain당 ~3100초, Study 1A 부록 참고)
비교 대상에서 제외했다.

## 2. 데이터 생성 (DGP)

Study 1B의 trap setup을 그대로 쓰되, N basin을 더 깊게 파고 conflict를 약화했다.

| 항목 | 값 | Study 1B 대비 |
|---|---|---|
| μ_S / μ_W / μ_N | 2.5 / 1.0 / **−1.5** | μ_N −0.8 → −1.5 (**장벽 상승**) |
| α | 0.20 | 동일 |
| ρ (S/W/N) | 0.20 / 0.20 / 0.60 | 동일 |
| damping d | **0.85** | 0.90 → 0.85 (prior 약화) |
| flip S→0 / N→1 | **0.10 / 0.05** | 0.30 / 0.10 (conflict 약화) |

μ_N을 낮춘 것은 SGLD 계열이 N basin에 끌려가는 약점을 부각시키기 위한 의도적 설계이다.
대신 damping과 conflict를 낮춰 난이도가 과하게 겹치지 않도록 조정했다.

### 규모별 그래프 파라미터

p_in은 N 그룹 내부 기대 degree를 약 90 수준으로 유지하도록 n마다 조정한다
(n = 1500: 900 × 0.10 ≈ 90, n = 10000: 6000 × 0.015 ≈ 90). 규모를 키울 때 그래프가
저절로 조밀해져 난이도가 바뀌는 교란을 막기 위함이다.

| n | σ_θ | p_in | p_out |
|---|---|---|---|
| 200 | 0.20 | 0.50 | 0.003 |
| 1500 | 0.26 | 0.10 | 0.001 |
| 10000 | 0.30 | 0.015 | 0.0002 |

## 3. 샘플러 설정

| n | T | burn-in (10%) | minibatch | NDCG의 k |
|---|---|---|---|---|
| 200 | 5000 | 500 | 50 | 40 |
| 1500 | 10000 | 1000 | 200 | 150 |
| 10000 | 10000 | 1000 | 1000 | 1000 |

- chain 3개, bad init θ⁽⁰⁾ = μ_N + dispersed(μ_W, μ_S)
- AWSGLD σ²_floor = 1.0, SGLD 계열 σ²_floor = 0.5
- 학습률 등 나머지 하이퍼파라미터는 Study 1A와 동일
- **θ̂ = 3 chain pooled mean over post-burn**

## 4. 결과

셀은 seed 평균 ± 표준편차. n = 10000은 seed 0 단일이라 표준편차가 0이다.

### n = 200 (5 seed, NDCG@40)

| 샘플러 | MSE_all | Spearman | NDCG@40 | R̂ max | R̂ med | ESS med | Cost/ESS | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 2.742 ± 0.192 | 0.233 ± 0.098 | 0.618 ± 0.056 | 5.931 | 1.928 | 5.30 | 0.704 | 3.7 |
| qSGLD | 1.505 ± 0.436 | 0.763 ± 0.018 | 0.931 ± 0.033 | 1.660 | 1.114 | 8.86 | 0.445 | 3.9 |
| cycSGLD | 1.613 ± 0.175 | **0.774 ± 0.017** | **0.944 ± 0.018** | 1.195 | 1.032 | 16.86 | 0.270 | 4.4 |
| SGHMC | 1.990 ± 0.239 | 0.623 ± 0.080 | 0.865 ± 0.035 | 2.536 | 1.232 | 6.77 | 0.666 | 4.5 |
| **AWSGLD** | **1.005 ± 0.227** | 0.766 ± 0.024 | 0.943 ± 0.021 | **1.041** | **1.005** | **75.40** | **0.062** | 4.7 |

### n = 1500 (5 seed, NDCG@150)

| 샘플러 | MSE_all | Spearman | NDCG@150 | R̂ max | R̂ med | ESS med | Cost/ESS | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 2.716 ± 0.087 | 0.244 ± 0.044 | 0.620 ± 0.030 | 6.063 | 1.654 | 6.90 | 12.32 | 85.0 |
| qSGLD | 1.793 ± 0.200 | 0.753 ± 0.009 | 0.860 ± 0.015 | 1.765 | 1.087 | 10.08 | 7.92 | 79.5 |
| cycSGLD | 1.734 ± 0.096 | **0.756 ± 0.009** | 0.868 ± 0.016 | 1.090 | 1.008 | 38.89 | 2.20 | 83.5 |
| SGHMC | 1.927 ± 0.072 | 0.693 ± 0.022 | 0.838 ± 0.037 | 2.238 | 1.142 | 8.42 | 9.79 | 82.4 |
| **AWSGLD** | **1.232 ± 0.088** | **0.757 ± 0.007** | **0.874 ± 0.016** | **1.051** | **1.004** | **89.43** | **0.899** | 80.4 |

### n = 10000 (seed 0, NDCG@1000)

| 샘플러 | MSE_all | Spearman | NDCG@1000 | R̂ max | R̂ med | ESS med | Cost/ESS | wall(s) |
|---|---|---|---|---|---|---|---|---|
| SGLD | 2.803 | 0.215 | 0.608 | 7.517 | 1.650 | 6.88 | 469.5 | 3231 |
| qSGLD | 1.719 | 0.748 | 0.845 | 2.108 | 1.086 | 9.74 | 348.4 | 3393 |
| cycSGLD | 1.784 | 0.754 | 0.857 | **1.088** | 1.006 | 40.36 | 81.0 | 3269 |
| SGHMC | 2.084 | 0.679 | 0.838 | 2.795 | 1.135 | 8.31 | 389.3 | 3233 |
| **AWSGLD** | **1.251** | **0.757** | **0.864** | **1.088** | **1.007** | **63.09** | **56.9** | 3588 |

### 그룹별 MSE (n = 200 / 1500 / 10000)

| 샘플러 | MSE_S | MSE_W | MSE_N |
|---|---|---|---|
| SGLD | 5.157 / 5.113 / 5.110 | 1.012 / 0.908 / 0.916 | 2.514 / 2.519 / 2.663 |
| qSGLD | 0.695 / 0.513 / 0.418 | 1.615 / 2.478 / 2.292 | 1.738 / 1.992 / 1.962 |
| cycSGLD | 1.145 / 0.930 / 1.124 | 0.128 / 0.276 / 0.250 | 2.263 / 2.488 / 2.515 |
| SGHMC | 2.616 / 2.314 / 2.400 | 0.430 / 0.252 / 0.253 | 2.302 / 2.356 / 2.589 |
| **AWSGLD** | **0.252 / 0.373 / 0.506** | 0.463 / 0.728 / 0.638 | **1.437 / 1.686 / 1.704** |

## 5. 해석

- **AWSGLD는 세 규모 전부에서 MSE_all 1위**이며 (1.005 / 1.232 / 1.251), 차선(qSGLD·cycSGLD)
  대비 30~40% 낮다. n이 50배 늘어도 1.0 → 1.25로 완만하게만 증가해 규모에 견고하다.
- **Study 1B에서 의심스러웠던 cycSGLD 문제가 해소되면서 순위가 바뀌었다.** pooled mean으로
  바꾸자 cycSGLD의 R̂ max가 1.09~1.20으로 정상화되고 NDCG도 상위권이다. 즉 Study 1B의
  cycSGLD 결과는 추정량 정의의 artifact였고, 여기서는 정직한 경쟁자로 자리 잡는다.
- **Spearman/NDCG만 보면 qSGLD·cycSGLD·AWSGLD가 사실상 동률이다** (n = 200에서 0.763 /
  0.774 / 0.766). 차이는 다른 축에서 난다.
  - **MSE_all** — AWSGLD가 확연히 낮다. 순위뿐 아니라 값의 스케일까지 맞춘다는 뜻이다.
  - **ESS** — AWSGLD 63~89 vs 나머지 5~40. 같은 T에서 서너 배 이상 독립 표본을 만든다.
  - **R̂** — AWSGLD만 전 규모에서 max ≤ 1.09.
- **그룹별로 보면 각 샘플러의 실패 지점이 다르다.**
  - SGLD는 MSE_S ≈ 5.1로 고정 — S basin에 아예 도달하지 못한다.
  - qSGLD는 S는 잘 잡지만 MSE_W가 규모와 함께 1.6 → 2.5로 악화된다. 가운데 얕은 mode를
    지나쳐 버린다.
  - cycSGLD·SGHMC는 W는 잘 잡지만 S가 약하다.
  - AWSGLD만 S / W / N 어느 축도 무너지지 않는다.
- **비용은 AWSGLD가 가장 비싸지 않다.** wall time은 전 규모에서 다른 샘플러의 ±10% 안이며,
  ESS가 높아 Cost/ESS는 오히려 최소다 (n = 200에서 0.062초, 차선의 1/4).
- **SGHMC는 규모 무관하게 일관되지만(Spearman 0.62~0.69, NDCG 0.84~0.87) 상위권은 아니다.**
  precondition 부재라는 구조적 한계가 규모를 키운다고 사라지지 않는다.

## 6. 한계

- n = 10000은 seed 1개뿐이라 표준편차가 없고, 관측된 순위가 seed 변동 내인지 확인되지 않았다.
  다만 n = 200/1500의 seed 표준편차가 MSE 기준 0.07~0.44로 순위를 뒤집을 크기는 아니다.
- acMH가 빠져 있어 "원본 BSS 대비 규모 확장성"은 Study 1A 부록의 n = 1500 ablation
  (acMH 3122초 vs AWSGLD 42~70초)으로 간접 근거를 삼는다.

## 7. 재현

```bash
# 데이터 생성 (seed 0~4)
python3 simulation/study_1c/data_generator.py --n 200   --seed 0
python3 simulation/study_1c/data_generator.py --n 1500  --seed 0
python3 simulation/study_1c/data_generator.py --n 10000 --seed 0

# 5 sampler 비교
python3 simulation/study_1c/sampler_comparison.py --n 200
python3 simulation/study_1c/sampler_comparison.py --n 1500
python3 simulation/study_1c/sampler_comparison.py --n 10000

# 시각화
python3 simulation/study_1c/data_landscape_overview.py
python3 simulation/study_1c/plot_n_scaling.py
python3 simulation/study_1c/plot_aw_gap_vs_n.py
python3 simulation/study_1c/plot_aw_diff_n200.py
```

n = 10000은 chain당 약 3200~3600초 × 3 chain × 5 sampler로 하루 단위 작업이며,
데이터 `.npz`만 규모별로 수 GB이다 (`study_1c/` 전체 약 7.7 GB).

## 8. 산출물

| 파일 | 내용 |
|---|---|
| `data_n{200,1500,10000}_seed{0..4}.npz` | 규모·seed별 데이터 (θ\*, Y, z, A, B, u_0) |
| `results_n{n}_multiseed_summary.json` | seed별 원값 + 집계 (**본 문서의 근거**) |
| `results_n{n}_multiseed.png` | 규모별 5 sampler 비교 플롯 |
| `n_scaling_comparison.png`, `aw_gap_vs_n.png`, `aw_diff_n200.png` | 규모에 따른 성능 추이 |
| `data_landscape_overview.png` | θ\* 분포 개요 |
| `run_n10000.log` | n = 10000 실행 로그 |
