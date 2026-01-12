# Multi-VAE Recommendation

Multi-VAE 모델을 활용하여 유저별 아이템 추천 리스트를 생성하는 프로젝트입니다.

---

## 📂 핵심 파일 요약

| 파일명 | 역할 |
| :--- | :--- |
| **`run_train.py`** | 전체 실행 메인 스크립트 (학습 및 추론 제어) |
| **`models.py`** | VAE 모델 아키텍처 정의 (MultiVAE, DeepMultiVAE) |
| **`preprocessing.py`** | 원본 CSV 데이터를 희소 행렬(`.npz`)로 변환 및 ID 매핑 |
| **`inference.py`** | 최종 제출 파일 및 2차 모델용 후보군 생성 |
| **`utils.py`** | 데이터 분할, 결과 저장 등 공통 유틸리티 함수 |

---

## 🚀 실행 순서

### 1. 라이브러리 설치
가상환경 활성화 후 아래 명령어를 입력하여 필수 패키지를 설치합니다.
```
pip install torch numpy pandas scipy
```

### 2. 데이터 전처리 (최초 1회)
학습용 데이터셋(`train_matrix.npz`)과 ID 매핑 파일(`mapping.pkl`)을 생성합니다.
```
python preprocessing.py
```

### 3. 학습 및 결과 생성
모델을 학습하고 `./output/` 폴더에 최종 결과물을 생성합니다.
```
python run_train.py
```

---

## 📍 출력 결과 (./output/)

* **submission.csv**: 대회 제출을 위한 Top-10 추천 리스트
* **vae_candidates_nomask.csv**: CatBoost 등 Re-ranking 모델 학습용 후보군
* **vae_candidates_test.csv**: CatBoost 추론용(본 아이템 제외) 후보군

---

## 💡 참고 사항
* **모델 설정**: `run_train.py` 내에서 하이퍼파라미터를 변경할 수 있습니다.
* **Early Stopping**: 성능 개선이 없을 경우 자동으로 학습이 중단되며 최적의 모델이 저장됩니다.
* **데이터 및 상대 경로**:
    * 기본 데이터 파일은 저장소에 포함되어 있지 않습니다.
    * `preprocessing.py` 및 `run_train.py` 상단에 설정된 데이터 경로(`../data/train/`)를 본인의 환경에 맞게 직접 수정해야 정상적으로 동작합니다.