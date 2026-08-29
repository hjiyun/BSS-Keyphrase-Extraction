# study_1b의 "mixture"에 대한 정리 — 1D 삽화 vs 실제 n차원 에너지

> 작성일 2026-08-29. 이 문서는 "study_1b는 mixture 실험인가?"라는 질문을 코드로 추적한
> 결과와, 그로부터 이 `mixture/` 폴더가 왜 생겼는지를 기록한다.

## 1. 질문

study_1b는 `local_trap_landscape.py` / `landscape_only.png` 등에서 **3-mode mixture
posterior**(N/W/S 3개 basin, 뚜렷한 에너지 장벽)를 타깃으로 내세운다. 그렇다면
study_1b의 샘플러들은 정말 이 다봉 mixture 에너지를 항해하는가?

## 2. 코드 추적 결과 — 두 개의 서로 다른 에너지가 섞여 있었다

### (A) 1D mixture 삽화 — `E_mix` (log-sum-exp)

- 정의: [`local_trap_landscape.py`](../local_trap_landscape.py) `E_mix(theta, ...)`
  ```
  E_mix(θ) = −log[ ρ_S·e^(−E_S(θ)) + ρ_W·e^(−E_W(θ)) + ρ_N·e^(−E_N(θ)) ]
  ```
  같은 **스칼라 θ 하나**에 대해 prior mean만 μ_S/μ_W/μ_N 세 개로 바꿔 겹친 것.
  그래서 N/W/S 3개 basin과 장벽이 생긴다.
- 이 함수를 **import해서 실행하는 파일은 그림 스크립트 두 개뿐**이다:
  `local_trap_landscape.py`(자기 자신), `data_landscape_overview.py`.
  두 파일 모두 x축이 스칼라 θ(−2~3.5)인 **1D 곡선 그림**을 그린다.
  (산출물: `landscape_only.png`, `data_landscape_overview.png`)
- 즉 `E_mix`는 **"노드 하나가 3개 basin을 가지면 좋겠다"는 설계 의도를 1D로 그린 삽화**다.

### (B) 실제 n=400 샘플러가 항해한 에너지 — 단일 BSS (mixture 아님)

- [`acmh_vs_awsgld.py`](../acmh_vs_awsgld.py), `sgld_only.py`가 실제로 쓰는 에너지는
  `keyphrase_functions(_awsgld).gibbs_mh` → `posterior_energy`:
  ```
  U(θ) = −Σ[Y·log((1−α)π) + (1−Y)·log(1−(1−α)π)]  +  ‖B(θ−u_0)‖² / (2σ²)
  ```
  ([`code_JOC/keyphrase_functions_awsgld.py:76`](../../../code_JOC/keyphrase_functions_awsgld.py))
- prior는 **그래프 기반 단일 Gaussian `‖B(θ−u_0)‖²` 하나뿐**이다. 노드마다 3-혼합 prior를
  얹지 않는다. `grep logsumexp acmh_vs_awsgld.py sgld_only.py` → **없음**.
- 다봉은 오직 **데이터 설계**로만 유도하려 했다: θ*를 S/W/N 3그룹에서 뽑고, label
  conflict(S의 30%→Y=0, N의 10%→Y=1)로 그래프 결합과 라벨 신호를 충돌시킴.

## 3. 결론 — 원래 설계의 틈

- 위 (A) 삽화는 "3개 basin이 있으면 좋겠다"는 **의도**였지만,
- 실제 (B) n차원 구현은 그 혼합을 넣지 않고 **단일 BSS**로 갔다.
- 그리고 **study_a0**에서 이 단일 BSS `U(θ)`가 **볼록**임을 증명했다(고정 (α,σ²)에서
  Hessian 최소고유값 ≈ +0.88, 모든 점). **label conflict를 넣어도 볼록** → 실제로는
  다봉 트랩이 없었다.
- 그래서 study_a0가 study_1b의 데이터·사후분포를 "그대로 재사용"할 수 있었던 것이고,
  두 실험이 항해한 **n차원 에너지는 동일**하다.

정리하면:
- **"study_1b는 mixture다"** — 타깃 설계·1D 삽화 기준으로는 **맞다**.
- **"에너지가 a0와 같다"** — 샘플러가 실제 항해한 n차원 에너지 기준으로는 **맞다**.
- 둘 다 참이며, 그 사이의 틈이 이 폴더의 존재 이유다.

샘플러가 **진짜** log-sum-exp mixture 에너지를 항해하는 실험은 지금까지 `study_2`
(trap_consensus / trap_multimode)뿐이었는데, 그것은 실데이터(doc2098) 기반이다.

## 4. 이 폴더(`mixture/`)가 하는 일

위 틈을 메운다. **합성 S/W/N 데이터 위에 실제 n차원 log-sum-exp K-mode mixture
에너지를 얹어**, 샘플러가 진짜 다봉을 항해하게 만들고(=1D 삽화의 의도를 n차원으로 구현),
그 결과를 **study_a0와 동일한 지표 세트**로 평가한다.

- 에너지: `U_mix(θ) = −logsumexp_k(−U_k(θ))`,
  `U_k(θ) = −loglik(Y|θ,α) + ‖B(θ − u^(k))‖²/(2σ²)` (study_2 consensus 트랩의 합성 이식)
- mode 중심 `u^(k)`: 공통 신호(진짜 키워드=S∪W 노드)는 모든 모드에서 높이고,
  각 모드는 고유한 미끼(N 노드의 mode별 부분집합)를 높인다 → 한 모드에 갇히면 미끼에 오염,
  여러 모드를 평균내면 미끼가 상쇄되고 공통 키워드만 남는 구조.
- 평가 지표(= study_a0 규약): Spearman / MSE / NDCG@50 (θ* 기준) ·
  **split-R̂**(median/q95/max, 4연쇄) · ESS(4연쇄 mean) · **Lowest U / Reached**
  (cutoff = AWSGLD 정상상태 에너지). AWSGLD는 R̂·복원에 **π-가중(w=G[J])** 적용.
- 이제 U가 **진짜 다봉(비볼록)**이므로 Lowest U / Reached / mode 방문수가 비로소
  의미를 갖는다(study_a0의 볼록 U에서는 무의미했던 축).

산출물은 이 폴더의 `README.md`와 각 스크립트 상단 docstring에 정리한다.
