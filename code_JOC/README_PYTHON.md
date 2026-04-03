# Python Version of BSS Keyphrase Extraction

이 디렉토리의 R 코드가 Python으로 변환되었습니다.

## 파일 구조

- `keyphrase_functions.py`: 핵심 함수들 (MCMC, 그래프 생성, 평가 메트릭 등)
- `main_graph.py`: 메인 분석 스크립트
- `semeval_author_obs.py`: SemEval 데이터셋 처리 스크립트
- `long_article_with_truth.py`: 긴 문서 처리 스크립트
- `keyphrase_examples.py`: 예제 스크립트
- `requirements.txt`: 필요한 Python 패키지 목록

## 설치

```bash
pip install -r requirements.txt
```

## 주요 변경사항

### 라이브러리 매핑

- `quanteda` → `create_fcm()` 함수 (간단한 버전, 프로덕션에서는 nltk/spacy 사용 권장)
- `mvtnorm` → `scipy.stats.multivariate_normal`
- `invgamma` → `scipy.stats.invgamma`
- `pROC/ROCR` → `sklearn.metrics`
- `MASS` → `scipy.stats`
- `coda` → 수렴 진단은 간단한 버전으로 구현

### 주요 함수

1. **`graph_generate()`**: 텍스트 파일에서 그래프 생성
2. **`semi_keyphrase2()`**: 반지도 학습 키프레이즈 추출
3. **`gibbs_mh()`**: Gibbs 및 Metropolis-Hastings 샘플링
4. **`precision_recall_auc()`**: 정밀도, 재현율, AUC 계산
5. **`FDR_cutoff()`**: FDR 기준으로 컷오프 찾기

## 사용법

### 메인 분석 실행

```python
from keyphrase_functions import main_function2
import os

# 파일 리스트 로드
file_list = os.listdir("/path/to/pre_process/directory")
result = main_function2(k=5, file_list=file_list, start_idx=201, end_idx=500)
```

### 개별 문서 처리

```python
from keyphrase_functions import graph_generate, semi_keyphrase2
import numpy as np

# 그래프 생성
graph = graph_generate(0, file_list)

# 키프레이즈 추출
grid = (np.arange(10, 43) - 5) / np.arange(10, 43)
result = semi_keyphrase2(graph, k=5, grid=grid)
```

## 주의사항

1. **텍스트 처리**: `create_fcm()` 함수는 간단한 구현입니다. 프로덕션 환경에서는 nltk, spacy, 또는 gensim과 같은 전문 NLP 라이브러리를 사용하는 것을 권장합니다.

2. **파일 경로**: 스크립트 내의 파일 경로를 실제 환경에 맞게 수정해야 합니다.

3. **C++ 컴포넌트**: 원본 R 코드의 `Component_MCMC.cpp`는 Python으로 완전히 변환되지 않았습니다. 필요시 `gibbs_mh()` 함수를 최적화하거나 Cython으로 재구현할 수 있습니다.

4. **메모리**: 대용량 데이터셋의 경우 메모리 사용량을 주의하세요.

## 성능 최적화 제안

1. NumPy의 벡터화 연산 활용
2. 큰 행렬 연산에 `scipy.sparse` 사용 고려
3. 병렬 처리를 위한 `multiprocessing` 또는 `joblib` 사용
4. Cython으로 핵심 루프 최적화

## 문제 해결

문제가 발생하면:
1. 파일 경로가 올바른지 확인
2. 필요한 패키지가 모두 설치되었는지 확인
3. 입력 데이터 형식이 올바른지 확인

