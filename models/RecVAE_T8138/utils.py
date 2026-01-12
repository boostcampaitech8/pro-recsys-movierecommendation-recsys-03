import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def recall_at_k(actual, predicted, k=20):
    set_actual = set(actual)
    set_predicted = set(predicted[:k])
    if len(set_actual) == 0: return 0
    return len(set_actual & set_predicted) / min(len(set_actual), k)