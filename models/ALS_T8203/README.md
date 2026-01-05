# ALS Recommendation Model

이 디렉토리는 ALS(Alternating Least Squares) 모델을  
기존 프로젝트 코드를 수정하지 않고 독립적으로 실행하기 위한 패키지입니다.

Movielens 데이터셋을 기준으로  
학습, 평가, 제출 파일 생성까지 전 과정을 포함합니다.

---

## Model Description

ALS는 Implicit Feedback 환경에서 널리 사용되는  
행렬 분해(Matrix Factorization) 기반 추천 알고리즘입니다.

User–Item 상호작용 행렬을 저차원 잠재 공간으로 분해하여  
사용자와 아이템의 latent factor를 학습합니다.

- User–Item 행렬을 User factor와 Item factor 행렬로 분해
- Alternating Least Squares 방식으로 두 행렬을 번갈아가며 최적화
- Implicit feedback 특성을 반영하여 confidence-weighted loss 사용
- 대규모 희소 행렬에 대해 효율적인 학습 가능
- 사용자 취향과 아이템 특성을 잠재 벡터 공간에서 표현

본 구현에서는 `implicit` 라이브러리의 ALS를 사용하여  
대규모 추천 문제를 안정적으로 학습합니다.

---

## Directory Structure
```
models/ALS/
├── __init__.py
├── config.py
├── data.py
├── model.py
├── trainer.py
├── evaluate.py
├── make_submission.py
├── main.py
└── README.md
```
```
__init__.py - ALS 디렉토리를 Python 패키지로 인식하게 하기 위한 파일

config.py - 데이터 경로 및 ALS 하이퍼파라미터를 관리하는 설정 파일

data.py - 데이터 로딩, 인코딩 및 URM(User-Item Matrix) 생성을 담당

model.py - ALS 알고리즘 핵심 로직

trainer.py - Train 데이터로 ALS 모델을 학습하는 로직 구현

evaluate.py - Validation 데이터 기준 Recall@K, NDCG@K 평가 수행

make_submission.py - 전체 데이터 재학습 후 추천 결과를 CSV 파일로 생성

main.py - 데이터 로딩부터 학습, 평가, 제출 파일 생성을 한 번에 실행
```
---
## Execution
```
python -m models.als.main
```
## Submission
```
python -m models.als.make_submission
```
