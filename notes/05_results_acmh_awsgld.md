# acMH + AWSGLD 결과 분석

## 관찰
- α̂ 편향은 무시할 수준이다. (작은 α에서의 PU MLE 약한 underestimation은 허용 범위다.)
- MSE_θ는 oracle init 하에서 시나리오 간 평탄하다 → 정상 거동이다.
- AUC_kp / NMSE / 상관계수는 시나리오 난이도에 따라 단조 변화한다.

## 다음 단계
- SGLD 계열 3종 (vanilla SGLD, qSGLD, cycSGLD) 추가 구현이 필요하다.
