# BSS + AWSGLD: Bayesian Semi-supervised Keyphrase Extraction

Wang et al. (2023)의 BSS 키프레이즈 추출 프레임워크에서
acMH-within-Gibbs 샘플러를 **AWSGLD** (Adaptively Weighted SGLD)로
대체하여 다봉 사후분포에서의 탐색 효율을 개선한다.

## 구조
- `keyphrase_functions.py` : Wang et al. 원본 BSS 함수
- `keyphrase_awsgld.py` : AWSGLD 변형
- `Component_MCMC.cpp` : 원본 acMH-within-Gibbs (Rcpp)
- `notes/` : 분석·실험 로그
- `*.pdf` : 참고 논문

## 비교 샘플러
1. acMH-within-Gibbs (Wang et al. 2023, baseline)
2. SGLD (Welling & Teh 2011)
3. qSGLD
4. cycSGLD (Zhang et al. 2020)
5. AWSGLD (Liang et al. 2022, 본 연구의 제안 도입)

## 진행 상황
- [x] Wang et al. 코드 분석
- [x] Study 1A 시나리오 4종 설계
- [x] AWSGLD 변형 구현
- [x] Easy 시나리오 검증
- [ ] Moderate / Difficult / Sparse 실행
- [ ] Study 1B (수렴 진단)
- [ ] Study 1C (mini-batch)
- [ ] Hulth 실데이터 평가
