# qSGLD 도입 메모

## 정의
- quasi-SGLD: 노이즈 공분산을 사후분포 곡률에 맞춰 변형한 변형판이다.
- 외부 참조 구현 (`Adaptively-Weighted-Stochastic-Gradient-MCMC-main/global_optimization_of_functions/`)
  의 함수를 BSS 모델 쪽으로 이식한다.

## 적용
- step-size 스케줄: ε_t = a / (b + t)^γ
- 본 BSS 설정에서는 γ=0.55, a, b는 시나리오별 튜닝이 필요하다.
