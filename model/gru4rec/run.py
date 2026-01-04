import pandas as pd
from model import GRU4Rec, GRU4RecMovieSide
from preprocessing import preprocess
from train import train_model, create_submission, create_submission_sideinfo
from utils import create_val_items, save_logits
from create_sideinfo import build_item2genres, build_item2year_bucket
import torch

def __main__():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv("D:\\movierec\\data\\train\\train_ratings.csv")

    sequences, sequence_users, num_items, item2idx, idx2item, user2idx, idx2user = preprocess(df)
    model = GRU4RecMovieSide(num_items, 18, 8)
    train_sequences, val_items = create_val_items(sequences, k=1)
    train_model(model, train_sequences, val_items, num_items, item2idx, idx2item, user2idx, idx2user, n_epochs=20, batch_size=64, learning_rate=0.001)
    # create_submission(model, sequences, item2idx, idx2item, user2idx, idx2user, 10)
    # checkpoint = torch.load("best_model\\gru4rec_best.pt", map_location=device)

    # model.load_state_dict(checkpoint["model_state_dict"])
    # item2idx = checkpoint["item2idx"]
    # idx2item = checkpoint["idx2item"]
    # user2idx = checkpoint["user2idx"]
    # idx2user = checkpoint["idx2user"]
    #create_submission(model, sequences, item2idx, idx2item, user2idx, idx2user, 10)
    item2year_bucket = build_item2year_bucket(item2idx, num_items, device)
    item2genres, genre2idx, idx2genre = build_item2genres(item2idx)
    create_submission_sideinfo(model, sequences, item2idx, idx2item, user2idx, idx2user, item2genres, item2year_bucket)

if __name__ == "__main__":
    __main__()