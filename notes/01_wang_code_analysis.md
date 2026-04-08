# Wang et al. (2023) BSS 코드 분석 노트

## 핵심 모듈
- `Component_MCMC.cpp` : acMH-within-Gibbs 샘플러 코어 (Rcpp)
- `keyphrase_functions.py` / `Keyphrase_functions.R` : 전처리, TextRank, 후처리
- `main_graph.*` : 그래프 구성 (단어 노드, 엣지)
- `Long_article_with_truth.*` : 정답 라벨 포함 평가 파이프라인

## 모델 구조
- π = (1 − α) · sigmoid(θ)  ← logit link 정의
- θ : 잠재 점수 (latent score)
- α : missing/noise rate
- Y ~ Bernoulli(π)

## 샘플러
- 현행: acMH-within-Gibbs (Adaptive Component-wise MH)
- 한계: 다봉(multimodal) 사후분포에서 모드 간 이동이 느리다.
