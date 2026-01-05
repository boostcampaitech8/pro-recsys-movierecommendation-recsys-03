# TiSASRec for Movie Recommendation

TiSASRec (Time Interval Aware Self-Attention for Sequential Recommendation) 모델 구현

> **Note**: RecBole-TRM이 pip 설치가 불가능하여, TiSASRec 모델과 Time-Aware Transformer를 직접 구현하였습니다.

## 모델 소개

**TiSASRec**은 SASRec을 기반으로 시간 간격(Time Interval) 정보를 추가로 학습하는 Sequential Recommendation 모델입니다.

- **논문**: [Time Interval Aware Self-Attention for Sequential Recommendation](https://dl.acm.org/doi/10.1145/3336191.3371786) (WSDM 2020)
- **저자**: Jiacheng Li, Yujie Wang, Julian McAuley

### 핵심 특징

1. **Time Interval Encoding**: 아이템 간 시간 간격을 학습에 반영
2. **Relative Position**: 절대 위치뿐만 아니라 상대적 시간 관계도 고려
3. **Self-Attention**: Transformer 기반의 시퀀스 모델링

## 디렉터리 구조

```
TiSASRec_T8108/
├── config/
│   └── tisasrec.yaml          # 모델 하이퍼파라미터 설정
├── dataset/
│   └── ml_movie/              # RecBole atomic files
│       └── ml_movie.inter     # 변환된 상호작용 데이터
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── tisasrec.py        # TiSASRec 모델 구현
│   ├── modules/
│   │   ├── __init__.py
│   │   └── transformer_layers.py  # Time-Aware Transformer 구현
│   ├── data_converter.py      # 데이터 변환 스크립트
│   ├── train.py               # 학습 스크립트
│   └── inference.py           # 추론 및 submission 생성
├── output/
│   ├── checkpoints/           # 모델 체크포인트
│   └── submission.csv         # 제출 파일
├── requirements.txt
└── README.md
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 1. 데이터 변환

대회 데이터를 RecBole 형식으로 변환:

```bash
cd src
python data_converter.py --input ../../../data/train/train_ratings.csv --output ../dataset/ml_movie
```

### 2. 학습

```bash
cd src
python train.py --config ../config/tisasrec.yaml
```

주요 하이퍼파라미터 조정:

```bash
# 에포크 수 조정
python train.py --epochs 100

# 학습률 조정
python train.py --learning_rate 0.0001

# 시퀀스 길이 조정
python train.py --max_seq_length 100

# 시간 간격 단위 조정
python train.py --time_span 512
```

### 3. 추론 및 제출 파일 생성

```bash
cd src
python inference.py --checkpoint ../output/checkpoints/TiSASRec-ml_movie.pth --topk 10
```

## 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `epochs` | 200 | 학습 에포크 수 |
| `train_batch_size` | 256 | 배치 크기 |
| `learning_rate` | 0.001 | 학습률 |
| `hidden_size` | 64 | 임베딩/히든 차원 |
| `n_layers` | 2 | Transformer 레이어 수 |
| `n_heads` | 2 | Attention head 수 |
| `MAX_ITEM_LIST_LENGTH` | 50 | 최대 시퀀스 길이 |
| `time_span` | 256 | 시간 간격 인코딩 단위 (초) |
| `hidden_dropout_prob` | 0.5 | Hidden dropout 확률 |
| `attn_dropout_prob` | 0.5 | Attention dropout 확률 |

## 참고 자료

- [RecBole](https://github.com/RUCAIBox/RecBole) - 기본 추천시스템 프레임워크
- [RecBole-TRM](https://github.com/RUCAIBox/RecBole-TRM) - TiSASRec 원본 구현 참고
- [TiSASRec 원본 (TensorFlow)](https://github.com/JiachengLi1995/TiSASRec)
- [논문](https://dl.acm.org/doi/10.1145/3336191.3371786)

## 구현 참고사항

이 구현은 RecBole-TRM의 TiSASRec을 참고하여 RecBole 기본 라이브러리 위에서 동작하도록 재구현하였습니다.

주요 구현 파일:
- `src/models/tisasrec.py`: TiSASRec 모델 (RecBole의 `SequentialRecommender` 상속)
- `src/modules/transformer_layers.py`: Time-Aware Multi-Head Attention, Transformer Encoder

## 대회 정보

- **평가 지표**: Recall@10
- **제출 형식**: 각 유저당 10개 아이템 추천 (user, item 컬럼)
