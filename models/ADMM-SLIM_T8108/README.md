# ADMM-SLIM for Movie Recommendation

RecBole 라이브러리를 활용한 ADMM-SLIM 모델 기반 영화 추천 시스템입니다.

## 모델 개요

**ADMM-SLIM (Alternating Direction Method of Multipliers - Sparse Linear Methods)**

- Item-Item 유사도 행렬을 학습하는 선형 모델
- L1 정규화로 sparse한 관계만 학습 (과적합 방지)
- ADMM 알고리즘으로 효율적인 최적화
- 논문: [ADMM SLIM: Sparse Recommendations for Many Users](https://doi.org/10.1145/3336191.3371774)

### 이 대회에 적합한 이유

1. **Item-Item 공존 구조 활용**: 순서 무관 + 랜덤 제거 문제에서 sequential 모델보다 유리
2. **Sparse 학습**: 노이즈에 강하고 일반화 성능 우수
3. **효율성**: Closed-form solution으로 빠른 학습

## 디렉터리 구조

```
ADMM-SLIM_T8108/
├── config/
│   ├── dataset.yaml       # 데이터셋 설정
│   └── model.yaml         # 모델 하이퍼파라미터
├── data/
│   └── movielens/         # RecBole 형식 데이터
│       └── movielens.inter
├── experiments/           # 실험 결과 저장
│   └── submission.csv
├── convert_data.py        # 데이터 변환 스크립트
├── run.py                 # 학습 실행
├── inference.py           # 추론 및 제출 파일 생성
├── tune.py                # 하이퍼파라미터 튜닝
├── utils.py               # 유틸리티 함수
├── requirements.txt       # 의존성
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 변환

원본 데이터를 RecBole 형식으로 변환합니다.

```bash
python convert_data.py
```

### 2. 모델 학습

```bash
# 기본 설정으로 학습
python run.py

# 하이퍼파라미터 지정
python run.py --lambda1 5.0 --lambda2 1000.0 --alpha 0.5
```

### 3. 하이퍼파라미터 튜닝

```bash
python tune.py
```

### 4. 추론 및 제출 파일 생성

```bash
python inference.py --output experiments/submission.csv
```

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `lambda1` | 5.0 | L1 정규화 강도 (sparsity 제어) |
| `lambda2` | 1000.0 | L2 정규화 강도 |
| `alpha` | 0.5 | Item popularity 기반 정규화 가중치 |
| `rho` | 10000.0 | ADMM 수렴 파라미터 |
| `k` | 100 | ADMM iteration 횟수 |
| `positive_only` | True | 양수 값만 사용 |
| `center_columns` | False | 열 중심화 여부 |

## 실험 결과

| 실험 | lambda1 | lambda2 | Recall@10 |
|------|---------|---------|-----------|
| Baseline | 5.0 | 1000.0 | - |
| ... | ... | ... | ... |

## 참고

- [RecBole Documentation](https://recbole.io/docs/)
- [ADMM-SLIM Paper](https://doi.org/10.1145/3336191.3371774)
- [RecBole ADMMSLIM Source](https://github.com/RUCAIBox/RecBole/blob/master/recbole/model/general_recommender/admmslim.py)
