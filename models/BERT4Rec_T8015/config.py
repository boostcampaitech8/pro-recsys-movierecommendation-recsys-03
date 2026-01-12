import torch
import os

class Config:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.TRAIN_DATA_PATH = os.path.abspath(os.path.join(base_path, "../../data/train/train_ratings.csv"))
        self.OUTPUT_DIR = os.path.abspath(os.path.join(base_path, "../../data/eval/BERT4Rec"))
        
        # Model Hyperparameters
        self.MAX_LEN = 200
        self.HIDDEN_UNITS = 256
        self.NUM_BLOCKS = 2
        self.NUM_HEADS = 2
        self.DROPOUT = 0.2
        self.MASK_PROB = 0.15
        
        # Training Hyperparameters
        self.BATCH_SIZE = 128
        self.LR = 0.001
        self.NUM_EPOCHS = 200
        self.PATIENCE = 10
        self.SEED = 42
        self.NUM_WORKERS = 8
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.TOP_K = 10
        self.WANDB_PROJECT = "MovieRec_research"

config = Config()
