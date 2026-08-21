# Study 1B — 다봉 사후분포에서의 local trap 탈출

> 산출물 위치: `simulation/study_1b/`
> 결과 근거 파일: `ava_metric_summary.json`, `sgld_metric_summary.json`, `ndcg_summary.json`, `ava_results.npz`, `sgld_results.npz`

## 1. 목적

Study 1A는 "난이도가 높으면 누가 무너지는가"를 봤다면, Study 1B는 원인을 하나로 좁힌다.
**나쁜 초기값에서 출발했을 때 사후분포의 국소 최소점(local trap)을 실제로 빠져나오는가**를
직접 측정한다.

이를 위해 (a) 사후분포가 실제로 다봉임을 사전 검증하고, (b) 모든 chain을 가장 무거운
N basin에서 출발시켜 S/W basin으로 넘어가는지를 escape time과 basin 체류량으로 계량한다.

## 2. 데이터 생성 (DGP)

n = 400. `data_generator.py`, 기본 seed 0 (1~4도 `_archive/`에 생성됨).
분포 파라미터는 `local_trap_landscape.py`의 `PARAMS`를 단일 진실 원천으로 삼는다.

| 항목 | 값 |
|---|---|
| ρ (S/W/N) | 0.20 / 0.20 / 0.60 → 80 / 80 / 240개 |
| μ_S / μ_W / μ_N | 2.5 / 1.0 / **−0.8** |
| σ_θ | 0.26 |
| α (PU 잡음) | 0.20 |
| 그래프 | SBM, p_in = 0.40, p_out = 0.005 |
| damping d | **0.90** (Study 1A의 0.85보다 강한 graph prior) |
| label conflict | S의 30% 강제 Y=0 (24개) + N의 10% 강제 Y=1 (24개) = **48/400** |

생성 절차는 Study 1A와 같으나 두 가지가 결정적으로 다르다.

1. **강한 그래프 대비** — p_in/p_out = 80배로 cluster가 뚜렷하다. graph prior가 노드를
   같은 cluster로 강하게 끌어당긴다.
2. **label conflict 주입** — PU 잡음 위에 라벨을 강제로 뒤집는다. 그래프는 "이 노드들은 같은
   그룹"이라고 말하는데 Y는 "아니다"라고 말하는 frustration이 생기고, 이 충돌이 joint
   posterior를 다봉으로 만든다.

`u_0 = B⁻¹(1 − d)**1**`을 그대로 쓰므로 oracle 초기화가 아니다 (CLAUDE.md 금지 사항).

### 타깃 분포와 장벽 설계

1D mixture 형태로 쓰면 다음과 같다 (log-sum-exp mixture, 가중합 금지).

```
E(θ) = −log[ ρ_S e^(−E_S(θ)) + ρ_W e^(−E_W(θ)) + ρ_N e^(−E_N(θ)) ]
```

μ_N을 Study 1A Easy의 −2.5에서 −0.8까지 끌어올려 N–W 장벽을 낮췄다. 의도는
"SGLD는 5000 스텝 안에 거의 못 넘고 AWSGLD는 넘는" 구간을 만드는 것이다.
장벽이 너무 높으면 전부 실패해 비교가 무의미해지고, 너무 낮으면 전부 성공해 역시 무의미하다.

다봉성은 `_archive/verify_multimodal.py`가 다중 초기값 경사하강으로 사전 검증했다.

### 다봉은 어떻게 유도되는가 (임의로 섞은 것이 아님)

다봉성은 인위로 부여한 것이 아니라 **BSS 사후분포 자체에서 유도**된다. BSS 사후는 두 힘의 곱이다.

```
p(θ | Y, α, A) ∝ likelihood(Y | θ, α) × prior(θ | A)
              = Π (1−α)π_i^{y_i}(1−(1−α)π_i)^{1−y_i} × exp(−‖B(θ−u_0)‖²/(2σ²))
```

- **likelihood**: 관측 라벨 Y가 θ를 끌어당긴다 (Y_i=1 → θ_i↑, Y_i=0 → θ_i↓).
- **prior**: 그래프 B = I − d·(D⁻¹A)ᵀ 가 **연결된 노드끼리 θ를 비슷하게** 만든다 (graph smoothness).

이 두 힘이 **충돌하도록** 데이터를 설계하면 사후분포에 여러 극대점(mode)이 생긴다 (frustration).
Study 1B는 두 장치로 이 충돌을 만든다.

1. **강한 그래프 결합** (p_in=0.40, p_out=0.005, damping=0.90) — prior가 "같은 cluster는 같은 θ"라고
   **강하게** 주장한다.
2. **label conflict 48/400** — cluster 안의 일부 라벨을 강제로 뒤집는다. 그래프는 "같은 cluster(같은 θ)"라
   하는데 뒤집힌 Y는 "반대"라고 말한다.

한 cluster 안에 라벨이 섞이면 사후분포는 **두 타협점**을 갖는다.

- **mode A ("라벨을 믿자")**: 뒤집힌 노드까지 라벨대로 θ 배치 → likelihood 만족, prior 위반.
- **mode B ("그래프를 믿자")**: cluster 전체를 같은 θ로 → prior 만족, likelihood 위반.

두 타협점 사이에 **에너지 장벽**이 생기고 이것이 곧 다봉이다. 즉 봉우리를 임의로 찍은 것이 아니라,
**모순된 증거(그래프 vs 라벨) 앞에서 사후분포가 자연히 갈라지는** 것이다.

`data_landscape_overview.png` (A) 패널의 에너지 곡선은 이 사후 에너지 `U(θ) = −log p(θ|Y,α)`를
위 **log-sum-exp 혼합으로 직접 계산**한 것이며, 세 골짜기(N −0.82 / W +1.02 / S +2.51)는
수치적으로 검출한 국소최소다. (B) 패널의 실제 θ\* 분포가 이 골짜기 위치와 정렬됨을 보여, 정의한
에너지 지형과 실제 데이터가 일치함을 확인한다.

> **CLAUDE.md 규칙과의 연결** — "사후 에너지를 직접 가중합으로 계산 금지, log-sum-exp mixture 필수"는
> 바로 이 때문이다. 가중합으로 만들면 항상 단일 minimum이 생겨 '가짜 다봉'이 되지만, log-sum-exp
> 혼합은 각 그룹의 사후분포가 정당하게 결합돼 **진짜 다봉이 유도**된다.

## 3. 샘플러 설정

| 항목 | 값 |
|---|---|
| T | 5000 |
| burn-in | 500 |
| chain 수 | 3 (R̂ 계산용, dispersed init) |
| minibatch | 100 (n = 400의 25%) |
| **bad init** | chain 0은 모든 노드 θ⁽⁰⁾ = μ_N = −0.8, 나머지 chain은 μ_W / μ_S |
| AWSGLD σ²_floor | **1.0** (기본 0.5에서 상향 — sweep 결과 고정) |
| AWSGLD 분할 수 | M_REGIONS = 1000 |

θ̂는 chain 0의 post-burn 평균이다 (Study 1C에서는 3 chain pooled로 바꾼다 — 아래 해석 참고).

`ava_metric_summary.json`의 `bad_init` 설명 문자열에는 `mu_N = -1.0`이 남아 있으나,
같은 파일의 `mu_map`이 `N: -0.8`이므로 실제 사용값은 **−0.8**이다 (문자열이 갱신 누락).

## 4. 평가 지표

- **MSE_all / MSE_g** — 전체 및 S/W/N 그룹별 (θ̂ − θ\*)² 평균.
- **Spearman** — θ̂ vs θ\* 순위 상관.
- **NDCG@k** — Study 1A와 동일 정의 (θ\* 순위를 [0,1]로 정규화한 graded relevance).
- **escape time** — θ_i가 자기 그룹 mode의 ±0.5 안에 처음 들어온 스텝. 못 들어오면 미탈출.
- **basin 체류량** — post-burn 구간에서 θ_i가 각 basin(±0.5) 안에 머문 (노드 × 스텝) 수.
- **R̂** — Gelman-Rubin, 노드별로 계산해 median / q90 / max를 본다.
- **ESS** — Geyer initial positive sequence, chain 0 post-burn 기준 노드별 중앙값.
- **Cost/ESS** — chain 0 wall time ÷ ESS median. 정보 1단위당 시간 비용.

## 5. 결과

### 5.1 복원 성능 (seed 0, chain 0)

| 샘플러 | MSE_all | MSE_S | MSE_W | MSE_N | Spearman | NDCG@20 | NDCG@80 |
|---|---|---|---|---|---|---|---|
| acMH | 2.282 | 2.626 | **0.105** | 2.893 | 0.026 | 0.518 | 0.547 |
| SGLD | 2.620 | 10.018 | 2.007 | **0.358** | 0.166 | 0.481 | 0.584 |
| qSGLD | 2.068 | 0.803 | 0.927 | 2.869 | 0.648 | **0.774** | **0.828** |
| cycSGLD | **1.272** | 3.118 | 0.099 | 1.048 | 0.692 | 0.711 | 0.800 |
| SGHMC | 1.704 | 6.661 | 0.557 | 0.433 | 0.390 | 0.665 | 0.725 |
| **AWSGLD** | 1.382 | **1.110** | 0.507 | 1.763 | **0.697** | 0.764 | 0.822 |

SGHMC의 NDCG는 저장된 `ndcg_summary.json`에 없어(`_archive/ndcg_at_k.py`의 대상 목록에서
누락) `sgld_results.npz`의 체인에서 동일 정의로 재계산했다.

### 5.2 수렴·효율

| 샘플러 | R̂ median | R̂ q90 | R̂ max | ESS median | wall(s) | Cost/ESS(s) |
|---|---|---|---|---|---|---|
| acMH | 1.116 | 1.361 | 1.777 | 7.50 | 170.85 | 22.77 |
| SGLD | 3.922 | 5.637 | 10.211 | 5.66 | 2.39 | 0.423 |
| qSGLD | 1.049 | 1.154 | 1.372 | 12.05 | 4.01 | 0.333 |
| cycSGLD | 3.859 | 4.533 | 5.089 | 4.72 | 3.97 | 0.841 |
| SGHMC | 2.101 | 2.926 | 4.495 | 7.09 | 3.51 | 0.495 |
| **AWSGLD** | **1.018** | **1.060** | **1.149** | **25.21** | 3.03 | **0.120** |

ESS와 Cost/ESS는 현재 `.npz` 체인 + 현재 요약 JSON의 wall time으로 **일관되게 재계산한 값**이다.
`_archive/extra_metrics_summary.json`에 저장된 Cost/ESS는 더 이른 실행의 wall time을 써서
qSGLD 0.21 / cycSGLD 0.51로 기록되어 있으나, ESS 자체는 동일하다.

### 5.3 trap 탈출 (bad init θ⁽⁰⁾ = μ_N)

**S basin(가장 먼 목표)에 도달한 노드 수 / 80, 괄호는 median escape step**

| 샘플러 | S 탈출 | W 탈출 | N 유지 |
|---|---|---|---|
| acMH | **0 / 80** (—) | 80 / 80 (638) | 240 / 240 (0) |
| SGLD | **0 / 80** (—) | 42 / 80 (1918) | 240 / 240 (0) |
| qSGLD | 80 / 80 (29) | 80 / 80 (2) | 239 / 240 (0) |
| cycSGLD | 4 / 80 (2484) | 80 / 80 (969) | 240 / 240 (0) |
| SGHMC | 15 / 80 (2761) | 80 / 80 (494) | 240 / 240 (0) |
| **AWSGLD** | **80 / 80 (17)** | 80 / 80 (2) | 238 / 240 (0) |

**basin 체류량 (노드 × post-burn 스텝, 총 400 × 4500 = 1,800,000)**

| 샘플러 | S | W | N | basin 밖 |
|---|---|---|---|---|
| acMH | 0 | 1,590,376 | 361 | 209,263 |
| SGLD | 46 | 91,513 | 890,997 | 817,444 |
| qSGLD | 305,942 | 649,927 | 121,085 | 723,046 |
| cycSGLD | 206 | 731,860 | 103,426 | 964,508 |
| SGHMC | 16,602 | 357,808 | 608,377 | 817,213 |
| **AWSGLD** | 168,851 | 775,839 | 112,884 | 742,426 |

## 6. 해석

- **AWSGLD는 R̂ / Spearman / ESS / Cost-per-ESS에서 1위**이며, 특히 R̂ max 1.15로 유일하게
  전 노드가 수렴 기준(< 1.1~1.2)을 만족한다. acMH 대비 Cost/ESS는 190배 효율이다.
- **탈출 능력이 세 부류로 갈린다.**
  - 완전 탈출: AWSGLD(median 17 step), qSGLD(29 step)
  - 부분 탈출: SGHMC(15/80, 2761 step), cycSGLD(4/80, 2484 step) — 넘긴 하지만 chain 후반이라
    post-burn 평균에 거의 기여하지 못한다
  - 미탈출: acMH, SGLD — S basin 도달 0개
- **acMH의 실패 양상이 특이하다.** 체류량을 보면 acMH는 post-burn의 88%를 W basin에서 보내고
  S와 N에는 사실상 가지 않는다. 세 그룹의 θ̂ 평균이 모두 0.88~0.90으로 거의 같다
  (S: 0.897, W: 0.897, N: 0.882). 즉 **모든 노드를 가운데 하나의 mode로 뭉개버렸고**, 그래서
  MSE_W만 0.105로 좋고 Spearman은 0.026으로 무의미하다. 국소 제안 기반 MH가 강한 graph
  prior 아래에서 mode 간 이동을 못 한 전형적 결과다.
- **cycSGLD의 MSE 1위(1.272)는 신뢰할 수 없다.** R̂ max 5.09로 chain이 수렴하지 않았고,
  S basin 도달은 4/80뿐이다. chain 0 하나만 우연히 좋았던 single-chain lucky 사례이며,
  이 문제 때문에 Study 1C에서는 θ̂ 정의를 3 chain pooled mean으로 바꿨다.
- **SGHMC는 vanilla SGLD를 명확히 개선한다** (MSE 2.62 → 1.70, R̂ max 10.21 → 4.50,
  Spearman 0.17 → 0.39). 모멘텀과 마찰이 minibatch 잡음을 상쇄하는 효과는 실재한다.
  그러나 precondition이 없어 (BᵀB) 기하를 못 쓰므로 qSGLD·AWSGLD 수준에는 못 미친다.
- **preconditioning이 탈출의 1차 요인, adaptive weighting이 2차 요인이다.**
  (BᵀB)⁻¹을 쓰는 qSGLD와 AWSGLD만 S basin에 완전 도달했다. 그 위에서 AWSGLD가
  qSGLD보다 나은 부분은 R̂(1.15 vs 1.37)과 ESS(25.2 vs 12.0), 즉 **탈출 이후의 혼합 품질**이다.

## 7. 재현

```bash
python3 simulation/study_1b/local_trap_landscape.py      # 타깃 분포 시각화 + 장벽 높이 출력
python3 simulation/study_1b/data_generator.py 0          # 데이터 생성 (seed 0)
python3 simulation/study_1b/data_landscape_overview.py 0 # θ* / Y 분포 개요
python3 simulation/study_1b/acmh_vs_awsgld.py 0          # acMH + AWSGLD (~10분, acMH가 대부분)
python3 simulation/study_1b/sgld_only.py 0               # SGLD / qSGLD / cycSGLD / SGHMC (~15초)
```

## 8. 산출물

| 파일 | 내용 |
|---|---|
| `data_seed0.npz` | θ\*, Y, Y_clean, conflict_mask, z, A, B, u_0, 파라미터 |
| `ava_results.npz` / `sgld_results.npz` | 전 chain의 θ 경로, θ̂, escape time, 노드별 R̂ |
| `ava_metric_summary.json` | acMH/AWSGLD 복원·체류·escape·R̂ |
| `sgld_metric_summary.json` | SGLD/qSGLD/cycSGLD/SGHMC 동일 항목 |
| `ndcg_summary.json` | NDCG@{10,20,50,80,160} (**SGHMC 누락**) |
| `sgld_trace_*.png`, `sgld_mode_visit.png`, `sgld_recovery.png` | trace / 체류 / 복원 플롯 |
| `data_landscape_overview.png`, `landscape_only.png` | 타깃 분포 |
| `_archive/` | σ² sweep, 다봉성 검증(GD), NDCG·ESS·혼동행렬 보조 스크립트, seed 1~4 데이터 |
