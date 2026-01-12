import os
import pickle
import time
import torch
import torch.nn as nn
import numpy as np
import wandb
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import DataLoader

from config import config
from utils import load_data, split_data, BERT4RecDataset, set_seed, get_data_mapping
from model import BERT4Rec

def train_epoch(model, dataloader, optimizer, criterion, config, scaler):
    model.train()
    total_loss = 0.0
    
    dataloader = tqdm(dataloader, desc="Training")
    
    for step, (seqs, labels) in enumerate(dataloader):
        seqs = seqs.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            logits = model(seqs)
            
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
            
            loss = criterion(logits, labels)
               
        scaler.scale(loss).backward()
        
        scaler.unscale_(optimizer)
        
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if step % 100 == 0:
            wandb.log({
                "train/total_loss": loss.item(),
                "train/learning_rate": optimizer.param_groups[0]['lr'],
                "train/gradient_norm": total_norm
            })
            dataloader.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

def evaluate(model, dataset, config):
    model.eval()
    hits = 0
    ndcg = 0
    count = 0
    
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    dataloader = tqdm(dataloader, desc="Evaluating")
    
    all_item_indices = torch.arange(1, model.item_num + 1).to(config.DEVICE)
    mask_token = model.item_num + 1
    
    val_seqs = dataset.user_seq
    
    num_batches = (len(val_seqs) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
    
    with torch.no_grad():
        for i in range(num_batches):
            start_idx = i * config.BATCH_SIZE
            end_idx = min((i + 1) * config.BATCH_SIZE, len(val_seqs))
            
            batch_seqs = val_seqs[start_idx:end_idx]
            
            input_seqs = []
            target_ids = []
            
            for seq in batch_seqs:
                target = seq[-1]
                target_ids.append(target)
                
                history = seq[:-1]
                if len(history) > config.MAX_LEN - 1:
                    history = history[-(config.MAX_LEN - 1):]
                
                pad_len = config.MAX_LEN - 1 - len(history)
                input_seq = [0] * pad_len + history + [mask_token]
                input_seqs.append(input_seq)
            
            input_ids_tensor = torch.LongTensor(input_seqs).to(config.DEVICE)
            target_ids_tensor = torch.LongTensor(target_ids).to(config.DEVICE)
            
            scores = model.predict(input_ids_tensor, all_item_indices)
            
            _, indices = torch.topk(scores, k=config.TOP_K, dim=-1)
            pred_items = indices + 1
            
            for k_idx in range(len(target_ids)):
                target = target_ids[k_idx]
                if target in pred_items[k_idx]:
                    hits += 1
                    rank = (pred_items[k_idx] == target).nonzero(as_tuple=True)[0].item()
                    ndcg += 1 / np.log2(rank + 2)
                    
                count += 1
                
    recall = hits / count if count > 0 else 0.0
    ndcg_score = ndcg / count if count > 0 else 0.0
    return recall, ndcg_score

def generate_submission(model, user_seqs, item_num, user2idx, id2item, output_file="submission.csv"):
    model.eval()
    
    sample_sub_path = os.path.join(os.path.dirname(config.TRAIN_DATA_PATH), "../eval/sample_submission.csv")
    sample_df = pd.read_csv(sample_sub_path)
    target_users = sample_df['user'].unique()
    
    user_preds = []
    item_preds = []
    
    target_user_indices = []
    valid_target_users = []
    
    for u in target_users:
        if u in user2idx:
            target_user_indices.append(user2idx[u])
            valid_target_users.append(u)
    
    batch_size = config.BATCH_SIZE
    num_batches = (len(target_user_indices) + batch_size - 1) // batch_size
    
    all_item_indices = torch.arange(1, item_num + 1).to(config.DEVICE)
    mask_token = item_num + 1
    
    print("Generating predictions...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Submission"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(target_user_indices))
            
            batch_users = target_user_indices[start_idx:end_idx]
            batch_seqs = [user_seqs[u] for u in batch_users]
            
            padded_seqs = []
            for seq in batch_seqs:
                if len(seq) > config.MAX_LEN - 1:
                    history = seq[-(config.MAX_LEN - 1):]
                else:
                    history = seq
                    
                pad_len = config.MAX_LEN - 1 - len(history)
                input_seq = [0] * pad_len + history + [mask_token]
                padded_seqs.append(input_seq)
            
            input_ids = torch.LongTensor(padded_seqs).to(config.DEVICE)
            scores = model.predict(input_ids, all_item_indices)
            
            for j, seq in enumerate(batch_seqs):
                seen_items = set(seq)
                for x in seen_items:
                    if 1 <= x <= item_num:
                        scores[j, x-1] = -float('inf')
            
            _, indices = torch.topk(scores, k=10, dim=-1)
            pred_items = indices + 1 
            
            pred_items_cpu = pred_items.cpu().numpy()
            
            for k in range(len(batch_users)):
                u_id = valid_target_users[start_idx + k]
                predicted_ids = [id2item[idx] for idx in pred_items_cpu[k]]
                
                for item_id in predicted_ids:
                    user_preds.append(u_id)
                    item_preds.append(item_id)
                    
    submission = pd.DataFrame({'user': user_preds, 'item': item_preds})
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")


def min_max_scale(matrix):
    m_min = matrix.min(dim=1, keepdim=True)[0]
    m_max = matrix.max(dim=1, keepdim=True)[0]
    return (matrix - m_min) / (m_max - m_min + 1e-9)


def extract_logits(model, user_seqs, item_num, user2idx, item2idx, config, base_name):
    model.eval()
    
    target_user2idx, target_item2idx, target_idx2item = get_data_mapping(config.TRAIN_DATA_PATH)
    
    sample_sub_path = os.path.join(os.path.dirname(config.TRAIN_DATA_PATH), "../eval/sample_submission.csv")
    sample_df = pd.read_csv(sample_sub_path)
    target_users = sample_df['user'].unique()
    
    model_user_indices = []
    for u in target_users:
        if u in user2idx:
            model_user_indices.append(user2idx[u])
            
    batch_size = config.BATCH_SIZE
    num_batches = (len(model_user_indices) + batch_size - 1) // batch_size
    
    all_item_indices = torch.arange(1, item_num + 1).to(config.DEVICE)
    mask_token = item_num + 1
    all_logits = []
    
    print("Extracting logits...")
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Logit Extraction"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(model_user_indices))
            
            batch_users = model_user_indices[start_idx:end_idx]
            batch_seqs = [user_seqs[u] for u in batch_users]
            
            padded_seqs = []
            for seq in batch_seqs:
                if len(seq) > config.MAX_LEN - 1:
                    history = seq[-(config.MAX_LEN - 1):]
                else:
                    history = seq
                    
                pad_len = config.MAX_LEN - 1 - len(history)
                input_seq = [0] * pad_len + history + [mask_token]
                padded_seqs.append(input_seq)
            
            input_ids = torch.LongTensor(padded_seqs).to(config.DEVICE)
            scores = model.predict(input_ids, all_item_indices)
            all_logits.append(scores.cpu())
            
    full_logit_matrix = torch.cat(all_logits, dim=0)
    
    print("Saving logits...")
    output_dir = config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    save_path = os.path.join(output_dir, f"{base_name}.npy")
    np.save(save_path, full_logit_matrix.cpu().numpy())
    print(f"Logits saved to {save_path}")

    mapping_save_path = os.path.join(output_dir, f"{base_name}_mapping.pkl")
    mapping_data = {
        'user2idx': user2idx,
        'item2idx': item2idx
    }
    with open(mapping_save_path, 'wb') as f:
        pickle.dump(mapping_data, f)
    print(f"Mapping saved to {mapping_save_path}")

def main():
    set_seed(config.SEED)
    
    kst_now = datetime.now() + timedelta(hours=9)
    now = kst_now.strftime("%Y%m%d_%H%M%S")
    base_name = f"BERT4Rec_{now}(EMB{config.HIDDEN_UNITS}-LEN{config.MAX_LEN}-MASK{config.MASK_PROB})"
    
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    
    model_save_path = os.path.join(config.OUTPUT_DIR, f"{base_name}.pth")

    run_name = f"BERT4Rec-EMB{config.HIDDEN_UNITS}_LEN{config.MAX_LEN}_MASK{config.MASK_PROB}_EP{config.NUM_EPOCHS}"
    
    wandb.init(project=config.WANDB_PROJECT, config={
        "batch_size": config.BATCH_SIZE,
        "epochs": config.NUM_EPOCHS,
        "lr": config.LR,
        "embedding_dim": config.HIDDEN_UNITS,
        "num_layers": config.NUM_BLOCKS,
        "mask_prob": config.MASK_PROB
    }, name=run_name)
    
    print("Loading data...")
    user_seqs, item_num, user2idx, item2idx, id2user, id2item, pop_confidence = load_data(config.TRAIN_DATA_PATH)
    print(f"Data loaded. Users: {len(user_seqs)}, Items: {item_num}")
    
    train_seqs, val_seqs = split_data(user_seqs)
    
    train_dataset = BERT4RecDataset(train_seqs, max_len=config.MAX_LEN, item_num=item_num, mask_prob=config.MASK_PROB)
    val_dataset = BERT4RecDataset(val_seqs, max_len=config.MAX_LEN, item_num=item_num, mask_prob=0)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        persistent_workers=True
    )
    
    model = BERT4Rec(item_num, config).to(config.DEVICE)
    print("Model initialized.")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, betas=(0.9, 0.98))
    scaler = torch.cuda.amp.GradScaler()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    
    best_recall = 0.0
    early_stopping_counter = 0
    
    wandb.watch(model, log="all")
    
    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, config, scaler)
        val_recall, val_ndcg = evaluate(model, val_dataset, config)
        
        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{config.NUM_EPOCHS} "
              f"Time: {end_time - start_time:.2f}s "
              f"Loss: {train_loss:.4f} "
              f"Recall@{config.TOP_K}: {val_recall:.4f} "
              f"NDCG@{config.TOP_K}: {val_ndcg:.4f}")
              
        wandb.log({
            "epoch": epoch + 1,
            "train_loss_epoch": train_loss,
            f"val/recall": val_recall,
            f"val/ndcg": val_ndcg
        })
        
        if val_recall > best_recall:
            best_recall = val_recall
            early_stopping_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"Best Model Saved!")
        else:
            early_stopping_counter += 1
            print(f"Early Stopping Counter: {early_stopping_counter}/{config.PATIENCE}")
            
        if early_stopping_counter >= config.PATIENCE:
            print("Early Stopping Triggered.")
            break
            
    wandb.finish()
    
    print("Generating submission file...")
    if os.path.exists(model_save_path):
        model.load_state_dict(torch.load(model_save_path))
        print(f"Loaded best model from {model_save_path}")
            
    output_dir = config.OUTPUT_DIR
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    submission_file_name = f"{base_name}.csv"
    output_path = os.path.join(output_dir, submission_file_name)
        
    generate_submission(model, user_seqs, item_num, user2idx, id2item, output_file=output_path)
    extract_logits(model, user_seqs, item_num, user2idx, item2idx, config, base_name)

if __name__ == "__main__":
    main()
