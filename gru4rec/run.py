import pandas as pd
from model import GRU4Rec
from preprocessing import preprocess
from train import train_model, create_submission
from utils import recall_at_k_gru4rec_session_parallel, create_val_items
import torch

def __main__():
    df = pd.read_csv("D:\\movierec\\data\\train\\train_ratings.csv")
    sequences, sequence_users, num_items, item2idx, idx2item, user2idx, idx2user = preprocess(df)
    model = GRU4Rec(num_items)
    train_sequences, val_items = create_val_items(sequences, k=1)
    train_model(model, train_sequences, val_items, num_items, item2idx, idx2item, n_epochs=10, batch_size=64, learning_rate=0.001)
    create_submission(model, sequences, item2idx, idx2item, user2idx, idx2user, 10)

if __name__ == "__main__":
    __main__()