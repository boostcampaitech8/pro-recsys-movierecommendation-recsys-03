# EASE Model for Movie Recommendation

영화 추천 대회를 위한 EASE (Embarrassingly Shallow Autoencoders) 모델 구현입니다.

## 프로젝트 구조

```
models/EASE_T8108/
├── config/
│   └── ease_config.yaml      # 설정 파일
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # 데이터 로딩 및 전처리
│   ├── ease_model.py         # EASE, WeightedEASE 모델
│   ├── metrics.py            # 평가 메트릭
│   └── utils.py              # 유틸리티 함수
├── output/                   # 출력 디렉터리
├── run_train.py              # 학습 스크립트
├── run_inference.py          # 추론 스크립트
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 기본 EASE 학습 및 평가

```bash
# 기본 파라미터로 학습
python run_train.py

# 정규화 파라미터 변경
python run_train.py --reg_weight 1000

# 시간 분포 분석 포함
python run_train.py --analyze_time
```

### 2. Weighted EASE (시간 가중치 적용)

```bash
# 시간 가중치 EASE
python run_train.py --weighted --tau 30 --reg_weight 500

# Hard threshold 추가
python run_train.py --weighted --tau 30 --delta_t_threshold 60
```

### 3. 하이퍼파라미터 그리드 서치

```bash
python run_train.py --grid_search
```

### 4. 제출 파일 생성

```bash
# 기본 EASE로 제출 파일 생성
python run_inference.py --reg_weight 500

# Weighted EASE로 제출 파일 생성
python run_inference.py --weighted --tau 30 --reg_weight 500

# 저장된 모델 로드해서 제출
python run_inference.py --model_path ./output/ease_model.npy
```

## 주요 파라미터

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `--reg_weight` | EASE 정규화 파라미터 (λ) | 500.0 |
| `--weighted` | 시간 가중치 EASE 사용 | False |
| `--tau` | 시간 감쇠 파라미터 (초) | 30.0 |
| `--delta_t_threshold` | 시간 차이 hard cutoff (초) | None |
| `--valid_random_items` | 검증용 랜덤 홀드아웃 아이템 수 | 9 |
| `--valid_seq_items` | 검증용 순차 홀드아웃 아이템 수 | 1 |

## 모델 설명

### Basic EASE
- Item-item similarity 학습: `B = (X^T X + λI)^(-1) X^T X`
- 대각 요소 제로 제약으로 자기 자신 추천 방지
- Closed-form solution으로 빠른 학습

### Weighted EASE
- 시간 근접성 기반 co-occurrence 가중치 적용
- 가중치 함수: `w(Δt) = exp(-Δt / τ)`
- 짧은 시간 간격의 아이템 쌍에 높은 가중치
- 같은 노출 묶음 복원에 효과적

## 검증 전략

대회의 데이터 생성 방식을 모방:
- Sequential holdout: 마지막 1개 아이템
- Random holdout: 랜덤 9개 아이템
- 총 10개 아이템을 검증용으로 사용

## 출력

- `output/submission.csv`: 제출용 파일 (user, item)
- `output/ease_model.npy`: 학습된 모델 가중치
- `output/experiment_*.log`: 실험 로그
- `output/experiment_*_results.csv`: 실험 결과

## 참고 문헌

- [EASE Paper](https://arxiv.org/abs/1905.03375): Embarrassingly Shallow Autoencoders for Sparse Data
