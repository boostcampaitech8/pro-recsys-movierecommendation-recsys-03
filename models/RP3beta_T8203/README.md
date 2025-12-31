# RP3beta Recommendation Model

이 디렉토리는 RP3beta(Random Walk based Recommendation) 모델을  
기존 프로젝트 코드를 수정하지 않고 독립적으로 실행하기 위한 패키지입니다.

Movielens 데이터셋을 기준으로  
학습, 평가, 제출 파일 생성까지 전 과정을 포함합니다.

---

## Model Description

RP3beta는 사용자–아이템 상호작용 그래프에서  
**Random Walk 기반으로 Item–Item 유사도 행렬을 학습**하는  
Implicit Feedback 추천 알고리즘입니다.

- User → Item → User → Item 경로를 따라 확률적으로 이동
- Item popularity bias를 완화하기 위해 `beta` 파라미터 사용
- 유사도 점수에 지수 가중치를 적용하기 위해 `alpha` 파라미터 사용
- Top-K pruning을 통해 계산 효율성과 추천 품질을 동시에 확보

---

## Directory Structure

```text
models/rp3beta/
├── __init__.py
├── config.py
├── data.py
├── model.py
├── trainer.py
├── evaluate.py
├── make_submission.py
├── main.py
└── README.md

__init__.py - rp3beta 디렉토리를 Python 패키지로 인식하게 하기 위한 파일

config.py - 데이터 경로 및 RP3beta 하이퍼파라미터를 관리하는 설정 파일

data.py - 데이터 로딩, 인코딩 및 URM(User-Item Matrix) 생성을 담당

model.py - RP3beta 알고리즘 핵심 로직과 Item-Item 가중치 계산 구현

trainer.py - Train 데이터로 RP3beta 모델을 학습하는 로직 구현

evaluate.py - Validation 데이터 기준 AUC, Recall@K, NDCG@K 평가 수행

make_submission.py - 전체 데이터 재학습 후 추천 결과를 CSV 파일로 생성

main.py - 데이터 로딩부터 학습, 평가, 제출 파일 생성을 한 번에 실행
```
---
