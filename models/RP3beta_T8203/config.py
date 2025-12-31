import os

# ======================
# Path
# ======================
BASE_DATA_DIR = "../../data"
TRAIN_DIR = os.path.join(BASE_DATA_DIR, "train")
EVAL_DIR = os.path.join(BASE_DATA_DIR, "eval")
OUTPUT_DIR = "./output"

# ======================
# RP3beta Hyperparameters
# ======================
ALPHA = 0.8
BETA = 0.6
TOPK = 1200

# ======================
# Evaluation
# ======================
RECALL_K = 10
NDCG_K = 10
N_NEG = 100

SEED = 42