# SGLD 계열 변형 추가

## 추가 샘플러
1. Vanilla SGLD (Welling & Teh, 2011)
2. qSGLD (quasi-SGLD)
3. cycSGLD (cyclic step-size SGLD, Zhang et al. 2020)

## 도입 의도
- AWSGLD의 weight-adaptation 효과를 분리 검증한다.
- 단순 Langevin 계열과의 비교 baseline을 확보한다.
