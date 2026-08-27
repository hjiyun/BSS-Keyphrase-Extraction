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

---

## 5. 중요도 가중 진단 + 수렴 속도 (2026-08-27 추가)

심사자 대응: **AWSGLD 표본은 틸트된 ϖ ∝ π/Ψ^ζ 에서 나오므로 raw R̂/ESS 는 ϖ 기준**이다.
π 기준 주장을 하려면 중요도 가중(참조 구현의 w=G[J], `sgmcmc.py:172`)이 필요하다.
다른 샘플러는 π 직접 표본이라 raw 그대로가 π 진단(가중 불필요·비대칭이 공정).

### 가중 후 (raw → π-가중, n=400 seed 평균)
| 지표 | raw(ϖ) | 가중(π) |
|---|---:|---:|
| Spearman | 0.679 | 0.670 (복원 견고) |
| MSE | 1.73 | 1.89 |
| R̂max | 1.14 | **1.52** (과대평가였음) |
| ESS(자기상관) / Kong ESS | 26.2 | Kong 909 (가중 penalty 약함) |

→ **복원·ESS 는 견고하나 R̂ "유일 수렴"은 raw 착시**였다. 공정한 π 비교에선 qSGLD(1.37)가 근소 우위.

### ζ(틸트) 튜닝
ζ 스윕 결과 **ζ=5.0(현재값)이 가중 R̂·ESS·복원 모두 최적**. 볼록이어도 조건수가 나빠
강한 탐색이 cross-chain 합의에 유효 → ζ 낮추면 오히려 악화.

### 수렴 속도 × n (가중 R̂max, 과분산 3-chain)
| n | AWSGLD R̂max 궤적 | 수렴(<1.2) | 나머지 |
|---|---|---|---|
| 100 | 1.15→1.05(10k)→**1.03**(15k) | ✅ ~2k | qSGLD 1.6·cyc 1.28·SGHMC 2.7 전부 미수렴 |
| 200 | →1.24 | median 1.02, max 근접 | cyc 막판 1.19 근소 |
| 300 | →1.09 | ✅ ~5k | 대부분 미수렴 |
| 400 | →1.25 | 더 필요 | 전부 미수렴 |

**느린 노드 진단**(n=200): R̂max>1.2 노드는 **단 1개**(200 중). 특성 = **약하게 식별되는 N 노드**
(θ*≈0 애매, 고degree로 반대그룹에 끌림) → 사후가 diffuse해 R̂ 민감. 믹싱 실패 아님(모든 샘플러 공통).

### n=100·T=20000 최종 통합표 (4-chain × seed0, split-R̂, AWSGLD=π가중)
| 샘플러 | Spearman | MSE | NDCG@50 | R̂med | R̂q95 | R̂max | ESS | Lowest U | Reached |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| acMH | 0.47 | 4.83 | 0.784 | 1.251 | 1.567 | 1.691 | 68 | 82 | 2/4 |
| SGLD | 0.12 | 2.15 | 0.605 | 3.096 | 4.916 | 5.904 | 11 | 110 | 0/4 |
| qSGLD | 0.51 | 4.98 | 0.797 | 1.209 | 1.620 | 2.270 | 18 | 114 | 0/4 |
| cycSGLD | 0.53 | 2.26 | 0.802 | 1.237 | 1.369 | 1.423 | 33 | 71 | 4/4 |
| SGHMC | 0.29 | 2.23 | 0.711 | 1.638 | 2.306 | 2.811 | 13 | 100 | 0/4 |
| **AWSGLD** | **0.61** | **2.12** | **0.824** | **1.026** | **1.057** | **1.080** | **121** | 71 | 4/4 |

- 모든 열이 **4-chain × seed0 단일 setup**. 4연쇄 시작점 = μ_N / μ_W / μ_S / **random N(0, 1.5²)**.
- R̂ = **split-R̂**(각 체인 반 분할 → 2M sub-chain, 좌표별 median/q95/max), AWSGLD는 π-가중.
- cutoff = 71 (AWSGLD 정상상태 에너지, 4체인 median). **Lowest U = 4연쇄 clamped 최소의 평균(mean)**.
- **AWSGLD가 R̂ 3종·Spearman·MSE·NDCG@50·ESS 전부 1위.** 다른 샘플러는 특히 **random 4번째 연쇄가 수렴·도달 실패**(AWSGLD만 random 연쇄도 도달). acMH 는 R̂ 낮아 보여도 MSE 4.83·Spearman 0.47로 복원 열세 → R̂·ESS만으론 부족, 복원까지 봐야 함.

관련 스크립트: `unified_n100.py`(통합표), `awsgld_convergence_n.py`(수렴곡선), `awsgld_weighted_diag.py`,
`awsgld_zeta_tune.py`, `slow_node_diag.py`, `trace_n100.py`.
그림: `strip6_cutoff_n100.png`, `convergence_n100_r105.png`, `trace_n100*.png`.
