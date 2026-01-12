import os
import time
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import numpy as np
import wandb

from config import config
import utils as data_utils
from model import LightGCN

def main():
    wandb_logger = wandb.init(project=config.wandb_project, config={
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "decay": config.decay,
        "embedding_dim": config.embedding_dim,
        "num_layers": config.num_layers
    },
    name=f"LGCN-EMB{config.embedding_dim}_LY{config.num_layers}_BT{config.batch_size}_LR{config.lr}_DCY{config.decay}_EP{config.epochs}")

    train_df, val_df, num_users, num_items, user2id, item2id, id2user, id2item, side_info = data_utils.load_data(config.train_data_path, config.val_ratio)

    popularity_confidence = data_utils.get_popularity_confidence(train_df, num_items)
    
    train_user_items = train_df.groupby('user')['item'].apply(list).to_dict()
    val_user_items = val_df.groupby('user')['item'].apply(list).to_dict()
    
    train_dataset = data_utils.RecDataset(train_df, num_users, num_items, popularity_confidence, train=True)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    
    adj_matrix, num_total_nodes = data_utils.create_adjacency_matrix(train_df, num_users, num_items, side_info)
    adj_matrix = adj_matrix.to(config.device)
    
    model = LightGCN(num_users, num_items, num_total_nodes, config, adj_matrix)
    model = model.to(config.device)
    
    wandb.watch(model, log="all")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scaler = torch.cuda.amp.GradScaler()
    
    os.makedirs(config.output_dir, exist_ok=True)
    best_val_recall = 0
    patience = 0
    max_patience = 10
    best_model_path = os.path.join(config.output_dir, "LightGCN_best_model.pt")

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for users, pos_items, neg_items, confidence in pbar:
            users = users.to(config.device)
            pos_items = pos_items.to(config.device)
            neg_items = neg_items.to(config.device)
            confidence = confidence.to(config.device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss, bpr_loss, reg_loss = model.bpr_loss(users, pos_items, neg_items, confidence)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            wandb_logger.log({
                "train/total_loss": loss.item(),
                "train/bpr_loss": bpr_loss.item(),
                "train/reg_loss": reg_loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/gradient_norm": total_norm
            })
            
        print(f"Epoch {epoch+1}: Average Loss = {total_loss / len(train_loader):.4f}")
        
        model.eval()
        val_loss = 0
        val_recall = 0
        val_ndcg = 0
        val_users = list(val_user_items.keys())

        val_batch_size = 100
        val_num_batches = (len(val_users) + val_batch_size - 1) // val_batch_size
        
        with torch.no_grad():
            for i in range(val_num_batches):
                start_idx = i * val_batch_size
                end_idx = min((i + 1) * val_batch_size, len(val_users))
                batch_user_indices = val_users[start_idx:end_idx]
                
                batch_users_tensor = torch.LongTensor(batch_user_indices).to(config.device)
                
                scores = model.predict(batch_users_tensor)
                scores = scores.cpu().numpy()
                
                for idx, user_idx in enumerate(batch_user_indices):
                    if user_idx in train_user_items:
                        train_items = train_user_items[user_idx]
                        scores[idx][train_items] = -np.inf
                    
                    top_k_indices = np.argpartition(scores[idx], -config.top_k)[-config.top_k:]
                    sorted_top_k = top_k_indices[np.argsort(scores[idx][top_k_indices])[::-1]]
                    
                    actual_items = val_user_items[user_idx]
                    val_recall += data_utils.recall_at_k(actual_items, sorted_top_k, config.top_k)
                    val_ndcg += data_utils.ndcg_at_k(actual_items, sorted_top_k, config.top_k)
        
        avg_recall = val_recall / len(val_users)
        avg_ndcg = val_ndcg / len(val_users)
        
        print(f"Validation: Recall@{config.top_k} = {avg_recall:.4f}, NDCG@{config.top_k} = {avg_ndcg:.4f}")
        wandb_logger.log({
            "val/recall": avg_recall,
            "val/ndcg": avg_ndcg
        })

        if avg_recall > best_val_recall:
            best_val_recall = avg_recall
            patience = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience += 1
            print(f"EarlyStopping counter: {patience} out of {max_patience}")
            if patience >= max_patience:
                print("Early stopping triggered")
                break
        
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    results = []
    
    all_users_ids = torch.arange(num_users).long()
    full_user_interacted_items = {**train_user_items}
    for u, items in val_user_items.items():
        if u in full_user_interacted_items:
            full_user_interacted_items[u].extend(items)
        else:
            full_user_interacted_items[u] = items
            
    user_interacted_items = full_user_interacted_items
    
    test_batch_size = 100
    num_batches = (num_users + test_batch_size - 1) // test_batch_size
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Inference"):
            start_idx = i * test_batch_size
            end_idx = min((i + 1) * test_batch_size, num_users)
            
            batch_users = all_users_ids[start_idx:end_idx].to(config.device)
            
            scores = model.predict(batch_users)
            
            scores = scores.cpu().numpy()
            
            for idx, user_idx in enumerate(range(start_idx, end_idx)):
                if user_idx in user_interacted_items:
                    seen_items = user_interacted_items[user_idx]
                    scores[idx][seen_items] = -np.inf
            
            top_k = config.top_k
            ind = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
            
            top_k_scores = np.take_along_axis(scores, ind, axis=1)
            sorted_ind_in_top_k = np.argsort(top_k_scores, axis=1)[:, ::-1]
            top_k_items = np.take_along_axis(ind, sorted_ind_in_top_k, axis=1)
            
            for j, user_idx in enumerate(range(start_idx, end_idx)):
                original_user_id = id2user[user_idx]
                for item_idx in top_k_items[j]:
                    original_item_id = id2item[item_idx]
                    results.append([original_user_id, original_item_id])
                    
    output_df = pd.DataFrame(results, columns=['user', 'item'])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(config.output_dir, f"LightGCN_{timestamp}.csv")
    output_df.to_csv(output_path, index=False)
    
    print(f"Saved to {output_path}")
    wandb_logger.finish()

if __name__ == "__main__":
    main()
