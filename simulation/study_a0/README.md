# Study A0 — 원본 BSS 에너지 U(θ)에 대한 샘플러 분석 (mixture 아님)

## 목적
Study 1B와 동일한 실데이터·사후분포를 쓰되, **mixture가 아닌 원래의 BSS 에너지 함수**

```
U(θ) = −loglik(Y | θ, α) + ‖B(θ − u_0)‖² / (2σ²)
```

에 대해 6개 샘플러(acMH / SGLD / qSGLD / cycSGLD / SGHMC / AWSGLD)를 정밀 비교한다.
핵심 질문: **"BSS 사후분포에서 AWSGLD가 정말 잘 복원하고 잘 샘플링하는가?"**

## 핵심 결론
1. **U(θ)는 볼록(convex)이다.** 고정 (α, σ²)에서 Hessian 최소고유값 > 0 (모든 점에서 +0.88).
   → 데이터를 어떻게 바꿔도 로컬 트랩을 만들 수 없다. "최저 에너지 도달(min-U)"은 볼록
   그릇에서 의미 없는 지표.
2. 따라서 실력 축은 min-U가 아니라 **복원(MSE·Spearman·NDCG) · 수렴(R̂) · 효율(ESS)**.
3. 이 축들에서 **AWSGLD가 우위**: 유일하게 R̂<1.2로 수렴, ESS 최고(2~5배), 복원 상위.
4. **cut-off 실험**: 수렴 지표가 정한 에너지선(U≤312, R̂<1.2 수렴 후 정상상태 에너지
   312.2±11.9)에서 멈추면 — 도달한 샘플러들은 Lowest-U 축에서 동률, 진짜 차이는 ESS.

## 파일 안내

### 최종 분석 스크립트
| 스크립트 | 산출물 | 내용 |
|---|---|---|
| `energy_diagnostics.py` | energy_minU_strip.png, awsgld_partition_visit*.png | 공통 에너지 진단, AWSGLD partition 방문 |
| `energy_path.py` | energy_path_bands.png | 에너지 band 궤적 (탐색 폭) |
| `recovery_sampling_figs.py` | recovery_scatter.png, sampling_quality_bars.png, convergence_diag.png | θ̂ vs θ* 복원 + 4지표 막대 + 수렴진단 |
| `multiseed_1b.py` | multiseed_1b.csv/json | 멀티시드 평균 (MSE/Spearman/R̂/S도달/NDCG/시간) |
| `cyc_awsgld_bss.py` | (콘솔/로그) | cyc-AWSGLD를 BSS 에너지에서 테스트 (AWSGLD가 여전히 우위) |
| `convergence_cutoff.py` | convergence_cutoff.png/csv | running R̂ 기반 수렴 판정 |
| `cutoff_derivation.py` | cutoff_derivation.png/csv | **cut-off=312의 도출 근거 시각화** |
| `strip6_cutoff.py` | strip6_cutoff.png/csv | **6-sampler: Lowest U(cutoff 클램프) + ESS** |

### 중간 탐색 스크립트 (최종본에 통합됨)
`cutoff_energy_ess.py`, `reach_time_ess.py`, `strip_cutoff_ess.py`, `replot_strip6.py`

### 의존성
- `data_generator.py`, `local_trap_landscape.py`: 데이터 생성 (Study 1B와 동일 PARAMS)
- `data_seed0-4.npz`: 생성 데이터 (gitignore — `python3 data_generator.py 0 1 2 3 4`로 재생성)
- `_archive/extra_metrics.py`(ESS), `_archive/ndcg_at_k.py`(NDCG)
- 대용량 체인(ava_/sgld_results.npz)·원본 샘플러(acmh_vs_awsgld/sgld_only)는 `../study_1b` 참조
  (recovery_sampling_figs.py, multiseed_1b.py에서 사용)

## cut-off = 312 근거 (요약)
| 지표 | AWSGLD | cycSGLD |
|---|---|---|
| running R̂max < 1.2 수렴 | O ~3961회 | X 미수렴(3.2~3.8) |
| 수렴후 정상상태 에너지 (seed 0/1/2) | 302.0 / 328.9 / 305.6 → **312.2±11.9** | 자격 없음 |

→ cut-off는 사람이 고른 값이 아니라 **R̂<1.2로 수렴한 유일한 샘플러(AWSGLD)가 정착한 에너지**.
