import os
import torch

class Config:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.train_data_path = os.path.abspath(os.path.join(base_path, "../../data/train/train_ratings.csv"))
        self.output_dir = os.path.abspath(os.path.join(base_path, "../../data/eval"))
        
        self.embedding_dim = 128
        self.num_layers = 2
        self.batch_size = 2048
        self.lr = 0.001
        self.epochs = 100
        self.decay = 1e-4
        self.top_k = 10
        self.val_ratio = 0.2
        self.seed = 42
        self.num_workers = 4
        
        # wandb settings
        self.wandb_project = "MovieRec_research"

config = Config()
