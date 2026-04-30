# cycSGLD 도입 메모

## 정의
- cyclic step-size SGLD (Zhang et al. 2020) : step-size를 코사인 사이클로
  변동시켜 한 사이클 내에서 탐색-활용을 교번하는 변형이다.

## 적용
- 사이클 길이 = 총 iter / K (K=4~8 권장)
- 각 사이클 후반 30%에서만 샘플 채취 (활용 단계).

## SGLD 계열 검토 정리
- vanilla SGLD : 단순 Langevin baseline
- qSGLD : 곡률 보정형 baseline
- cycSGLD : 다봉 탐색 baseline
- AWSGLD : weight-adaptation 적용 (제안 도입)

cycSGLD까지 추가하여 SGLD 계열 baseline 4종 검토를 마무리한다.
초안 비교 연구는 `simlation/BSS_032926.tex`로 작성 중이다.
