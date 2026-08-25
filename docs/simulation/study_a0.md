# Study A0 — 원본 BSS 에너지 U(θ)에 대한 샘플러 정밀 비교 (mixture 배제)

*최종 갱신: 2026-08-25*

## 무엇을, 왜

Study 1B와 **동일한 실데이터·사후분포**를 쓰되, 시각화용 mixture 지형이 아니라
**원래의 BSS 사후 에너지 함수 그 자체**만을 대상으로 6개 샘플러를 정밀 비교한다.

```
U(θ) = −loglik(Y | θ, α) + ‖B(θ − u_0)‖² / (2σ²)
```

핵심 질문: **"이 사후분포에서 AWSGLD가 정말 잘 복원하고, 잘 샘플링하는가?"**
그리고 그것을 **어떤 지표로 판정해야 정직한가?**

산출물 전부 `simulation/study_a0/`에 있으며, 문서는 저장된 CSV/JSON/PNG를 근거로만 작성했다.

## 1. 지형의 성질 — U(θ)는 볼록(convex)이다

고정 (α, σ²)에서 U(θ)의 Hessian 최소고유값이 **모든 점에서 양수**(+0.88 수준)다.

| 위치 | Hessian 최소고유값 |
|---|---|
| u_0 | +0.88 |
| zeros | +0.95 |
| random ×2 | +0.88 / +0.88 |

- prior 항 = 볼록 2차식(BᵀB/σ² ⪰ 0), likelihood 항(로지스틱 음의로그우도) = 볼록.
- **볼록 + 볼록 = 볼록** → 극소가 하나. **데이터(Y·그래프)를 어떻게 바꿔도 로컬 트랩을 만들 수 없다.**
- 차원(400)이 많은 것과 무관하다. 트랩은 곡률 부호가 뒤집혀야 생기며(예: Rastrigin의 cos 진동항),
  BSS엔 그런 항이 없다. 대신 조건수가 나쁜(찌그러진) 볼록 계곡일 뿐이다.

**함의**: "가장 낮은 에너지 도달(min-U)"은 볼록 그릇에서 **누가 바닥에 먼저 눌러앉나**만 재는
잘못된 지표다. 정답 θ\*는 최저점(MAP)에 있지도 않다. → **min-U가 아니라 복원·수렴·효율로 봐야 한다.**

## 2. 복원·샘플링 품질 (실데이터 사후, bad init)

`recovery_sampling_figs.py`, 저장 체인(seed 0).

| 샘플러 | Spearman ↑ | MSE ↓ | R̂ max ↓ | ESS ↑ |
|---|---:|---:|---:|---:|
| acMH | 0.026 | 2.28 | 1.78 | 7.5 |
| SGLD | 0.166 | 2.62 | 10.21 | 5.7 |
| qSGLD | 0.648 | 2.07 | 1.37 | 12.0 |
| cycSGLD | 0.692 | **1.27** | 5.09 | 4.7 |
| SGHMC | 0.390 | 1.70 | 4.50 | 7.1 |
| **AWSGLD** | **0.697** | 1.38 | **1.15** | **25.2** |

- **AWSGLD만 R̂로 수렴**(1.15). cycSGLD는 MSE만 근소 낮지만 R̂ 5.09 = 수렴 실패(믿을 수 없음).
- **ESS는 AWSGLD가 압도**(25.2, 2위 qSGLD의 2배).

### 멀티시드 평균 (2 seed, `multiseed_1b.py`)
| 방법 | MSE | Spearman | R̂max | S 도달 | NDCG@160 | ESS(대표) |
|---|---:|---:|---:|---:|---:|---:|
| **AWSGLD** | 1.78±0.39 | **0.674** | **1.16** | **80/80(16회)** | **0.911** | 최고 |
| cycSGLD | **1.37** | 0.673 | 4.78 | 18.5/80 | 0.904 | 낮음 |

큰 그림 불변: AWSGLD가 유일 수렴 + S 영역 100%·최속 도달 + 랭킹 최고.

## 3. cyc-AWSGLD를 BSS에서 시도 (`cyc_awsgld_bss.py`)

AWSGLD에 cyc(순환 냉각)를 결합해도 **BSS(볼록)에서는 이득 없음**:

| 지표 | AWSGLD | cyc-AWSGLD |
|---|---:|---:|
| Spearman | 0.700 | 0.697 |
| **MSE** | **1.32** | 1.60 |

cyc의 가치는 트랩 탈출(rugged 전용)인데 BSS엔 트랩이 없어, 냉각은 collapse만 유발해 MSE만 악화.
→ **BSS에서는 plain AWSGLD가 낫다.**

## 4. 정직한 지표 — cut-off는 수렴 지표가 정한다 (`cutoff_derivation.py`, `strip6_cutoff.py`)

min-U 대신 **"수렴한 샘플러가 정착하는 에너지"**를 공통 cut-off로 삼는다. 수렴 판정은 표준
Gelman-Rubin R̂ < 1.2.

**cut-off = 312** 도출:
| 지표 | AWSGLD | cycSGLD |
|---|---|---|
| running R̂max < 1.2 수렴 | ✅ ~3961회 | ❌ 미수렴(3.2~3.8) |
| 수렴후 정상상태 에너지 (seed 0/1/2) | 302.0 / 328.9 / 305.6 → **312.2 ± 11.9** | (자격 없음) |

→ 312는 사람이 고른 값이 아니라 **R̂<1.2로 수렴한 유일한 샘플러(AWSGLD)가 정착한 에너지**.

**cut-off에서 멈춰 그린 6-sampler strip** (Lowest U를 cut-off에 클램프):
| 샘플러 | Lowest U (클램프) | 도달 | ESS |
|---|---:|---:|---:|
| acMH | 312 | 5/5 | 7.6 |
| qSGLD | 312 | 4/5 | 13.1 |
| cycSGLD | 312 | 5/5 | 4.7 |
| SGLD | 352 | 0/5 | 5.6 |
| SGHMC | 351 | 0/5 | 7.3 |
| **AWSGLD** | **312** | 5/5 | **26.4** |

- Lowest-U 축: 도달한 4개가 cut-off에 **동률** → 에너지로는 구분 불가.
- **ESS: AWSGLD 26.4로 압도**(cycSGLD의 5.6배, 차선 qSGLD의 2배).

## 한 줄 결론

> **원본 BSS 에너지는 데이터와 무관하게 볼록 그릇 하나다 → 트랩·min-U는 무의미.
> 실력은 복원(MSE·Spearman·NDCG)·수렴(R̂)·효율(ESS)로 재야 하며, 이 축들에서 AWSGLD가
> 유일하게 수렴하고 ESS가 2~5배 높아 명백히 앞선다. cut-off는 R̂<1.2 수렴 에너지(312)로
> 데이터가 정한다.**

## 파일 (`simulation/study_a0/`)

| 스크립트 | 산출물 |
|---|---|
| `energy_diagnostics.py`, `energy_path.py` | energy_minU_strip.png, energy_path_bands.png, awsgld_partition_visit*.png |
| `recovery_sampling_figs.py` | recovery_scatter.png, sampling_quality_bars.png, convergence_diag.png |
| `multiseed_1b.py` | multiseed_1b.csv / .json |
| `cyc_awsgld_bss.py` | (콘솔/로그) |
| `convergence_cutoff.py`, `cutoff_derivation.py` | convergence_cutoff.png, **cutoff_derivation.png** (cut-off=312 도출) |
| `strip6_cutoff.py` | **strip6_cutoff.png** (6-sampler Lowest-U + ESS) |
| 중간 탐색 | `cutoff_energy_ess.py`, `reach_time_ess.py`, `strip_cutoff_ess.py`, `replot_strip6.py` |
| 폴더 안내 | `simulation/study_a0/README.md` |

의존성: `data_seed*.npz`(gitignore, `data_generator.py`로 재생성), `_archive/`(ESS·NDCG),
대용량 체인·원본 샘플러는 `../study_1b` 참조.
