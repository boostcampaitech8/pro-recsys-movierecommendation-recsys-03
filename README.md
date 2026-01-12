# 🎞️ Movie Recommendation

---

# 💁🏼‍♂️ 대회 소개

MovieLens 데이터를 Implicit Feedback 데이터로 전처리한 데이터를 토대로 Recall@10를 올리는 태스크로 진행됩니다.
테스트 데이터는 특정 시점 이후의 데이터(Sequential)와 특정 시점 이전의 랜덤 샘플링된 데이터(Static)가 섞여있습니다.
따라서 대회의 목적에 맞게 Recall@10을 올리기 위해서는 Sequential Data와 Static Data를 모두 예측해야 합니다.

# 😀 팀 소개

| 구승민 | 박주연 | 송정호 | 이다검 | 이성재 | 최연우 |
| --- | --- | --- | --- | --- | --- |
| <a href = 'https://github.com/gsmin02'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/86878502?v=4'></a> | <a href = 'https://github.com/wndus0212'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/83656909?v=4'></a> | <a href = 'https://github.com/dynamite885'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/50672132?v=4'></a> | <a href = 'https://github.com/DaGoMi1'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/155869202?v=4'></a> | <a href = 'https://github.com/localman211'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/223304427?v=4'></a> | <a href = 'https://github.com/Choiyeonw00'><img  width="100" height="100" src = 'https://avatars.githubusercontent.com/u/105337438?v=4'></a> |

# 🖥️ 환경

### 하드웨어 환경

| **구분** | **상세 스펙** |
| --- | --- |
| **운영체제(OS)** | **Ubuntu 20.04.6 LTS (Focal Fossa)** |
| **GPU** | **NVIDIA Tesla V100-SXM2 (VRAM 32GB)** |
| **RAM** | **88 GiB (약 90GB)** |
| **저장 장치** | **SSD (총 100GB)** |

### 협업

| **구분** | **역할** |
| --- | --- |
| **Slack** | 회의록 관리 및 연락 |
| **GitHub** | PR을 통한 코드 품질 관리 |
| **Zoom** | 실시간 화상 회의 및 화면 공유를 통한 상세 논의 |
| **Notion** | 프로젝트 히스토리 및 서버 사용 스케줄 저장 |

# 🎯 주요 구현 기능

### 대회 기간 구현한 모델

| **VAE** | **RecVAE** | **Multi-Modal VAE** | Multi-VAE |  |  |
| --- | --- | --- | --- | --- | --- |
| **Public Score** | **0.1393** | **0.1318** | 0.1291 |  |  |
| **Sequential** | **BERT4Rec** | **gSASRec** | GRU4Rec | S3 Rec |  |
| **Public Score** | **0.1191** | **0.1154** | 0.0865 | 0.0886 |  |
| **Others** | **EASE** | LightGCN | ADMM-SLIM | MF - ALS | RP3beta |
| **Public Score** | **0.1608** | 0.1218 | 0.1573 | 0.1395 | 0.1116 |

### 최종 앙상블 선정 모델

| 모델 | **RecVAE** | **Multi-Modal VAE** | **BERT4Rec** | **gSASRec** | **EASE** |
| --- | --- | --- | --- | --- | --- |
| 비율 | 20% | 20% | 20% | 20% | 20% |

# 💯 최종 성적

**Public 순위 : 2위 / 7팀** (Recall@10 : 0.1867)

**Private 순위 : 1위 / 7팀** (Recall@10 : 0.1729)

# 📌 실행 방법

### 각 모델 학습 및 추론 (+ logit 추출)

```bash
python /models/EASE/EybridEASE.py --ITEM_BEST 660 --USER_BEST 4573 # EASE
python /models/RecVAE_T8138/run_train.py # RecVAE
python /models/M2VAE_TT8138/run_train.py # M2VAE
python /models/Multi-VAE_TT8138/run_train.py # Multi-VAE
python /models/gSASRec_T8015/main.py # gSASRec
python /models/BERT4Rec_T8015/main.py # BERT4Rec
python /models/LightGCN_T8015/main.py # LightGCN
python /models/GRU4RecF.py # GRU4RecF
```

### 앙상블

```python
# ensemble.py 내부에서 수정해서 실행

if __name__ == "__main__":
    ensemble(w_rvae=0.2, w_m2vae=0.2, w_bert=0.2, w_gsas=0.2, w_ease=0.2)
```

# 📋 파일 구조

- 디렉터리 구조
    
    ```
    C:.
    ├─code
    │  │  datasets.py
    │  │  inference.py
    │  │  models.py
    │  │  modules.py
    │  │  preprocessing.py
    │  │  README.md
    │  │  requirements.txt
    │  │  run_pretrain.py
    │  │  run_train.py
    │  │  sample_submission.ipynb    
    │  │  trainers.py
    │  │  utils.py
    │  │
    │  └─output
    ├─data
    │  ├─eval
    │  │      sample_submission.csv
    │  │
    │  └─train
    │          directors.tsv
    │          genres.tsv
    │          Ml_item2attributes.json
    │          titles.tsv
    │          train_ratings.csv
    │          writers.tsv
    │          years.tsv
    │
    ├─EDA
    │
    ├─models
    │   └─<model-name>_<camper-id>
    │          config.py
    │	         main.py
    │	         model.py
    │
    ├─ensembles
    |  |  ensemble.py
    |  |
    |  └─submit
    |          submission1.csv
    |          submission2.csv
    └─logits
         <model-name>_Logits.npy
         <model-name>_mapping.pkl
    ```
