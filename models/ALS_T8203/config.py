import os

# =========================
# Path
# =========================
BASE_DATA_PATH = "/content/drive/MyDrive/MovieRec/data"

TRAIN_PATH = os.path.join(BASE_DATA_PATH, "train")
EVAL_PATH = os.path.join(BASE_DATA_PATH, "eval")
OUTPUT_PATH = "./outputs/als"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# =========================
# ALS Hyperparameters
# =========================
ALS_FACTORS = 64
ALS_REGULARIZATION = 0.01
ALS_ITERATIONS = 400
ALS_RANDOM_STATE = 42

# =========================
# Evaluation
# =========================
TOP_K = 10