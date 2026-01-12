import numpy as np
import pandas as pd
import pickle
import os

def min_max_scale(logits):
    low = logits.min(axis=1, keepdims=True)
    high = logits.max(axis=1, keepdims=True)
    return (logits - low) / (high - low + 1e-9)

def ensemble(w_rvae=0.245, w_m2vae = 0.105, w_bert=0.1125, w_gsas=0.3375, w_ease=0.2):
    print("🚀 Start Ensemble ...")
    print(f"Weights -> RecVAE: {w_rvae}, M2VAE: {w_m2vae}, BERT4Rec: {w_bert}, gSASRec: {w_gsas}, EASE: {w_ease}")
    
    # 경로 설정
    data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    ensemble_dir = os.path.join(data_root, "../logits")
    
    # 로드
    print(f"📂 Loading logits...")
    recvae_logits = np.load(os.path.join(ensemble_dir, "RecVAE_Logits.npy"))
    m2vae_logits = np.load(os.path.join(ensemble_dir, "M2VAE_Logits.npy"))
    ease_logits = np.load(os.path.join(ensemble_dir, "EASE_Logits.npy"))
    bert_logits = np.load(os.path.join(ensemble_dir, "BERT4Rec_Logits.npy"))
    gsas_logits = np.load(os.path.join(ensemble_dir, "gSASRec_Logits.npy"))
    
    # 매핑 파일 로드
    with open(os.path.join(ensemble_dir, "RecVAE_mapping.pkl"), 'rb') as f:
        m_static = pickle.load(f) # 기준
    with open(os.path.join(ensemble_dir, "BERT4Rec_mapping.pkl"), 'rb') as f:
        m_bert = pickle.load(f)   # BERT 전용
    with open(os.path.join(ensemble_dir, "gSASRec_mapping.pkl"), 'rb') as f:
        m_gsas = pickle.load(f)   # gSASRec 전용

    # 유저 재배치 (Row)
    sample_sub_path = os.path.join(data_root, 'eval/sample_submission.csv')
    sample_df = pd.read_csv(sample_sub_path)
    seq_user_order = sample_df['user'].unique()
    
    s_u2idx = m_static['user2idx']
    u_target_idx = [] # 옮겨갈 위치 (RecVAE 순서)
    u_source_row = [] # 가져올 위치 (Sequential 모델 npy 행)
    
    for r, u_id in enumerate(seq_user_order):
        if u_id in s_u2idx:
            u_target_idx.append(s_u2idx[u_id])
            u_source_row.append(r)

    # 아이템 재배치 (Column)
    s_i2idx = m_static['item2idx']
    b_i2idx = m_bert['item2idx']
    g_i2idx = m_gsas['item2idx']
    
    # BERT4Rec Reorder Index
    b_reorder = []
    # gSASRec Reorder Index
    g_reorder = []
    
    # RecVAE item order 순회
    for iid, _ in sorted(s_i2idx.items(), key=lambda x: x[1]):
        # BERT Alignment
        if iid in b_i2idx:
            b_col = b_i2idx[iid] - 1
            b_reorder.append(b_col)
        else:
            print(f"Warning: Item {iid} not found in BERT mapping.")
            b_reorder.append(0)
            
        # gSASRec Alignment
        if iid in g_i2idx:
            g_col = g_i2idx[iid] - 1
            g_reorder.append(g_col)
        else:
            print(f"Warning: Item {iid} not found in gSASRec mapping.")
            g_reorder.append(0)

    # 재정렬 실행
    print("🔄 Re-aligning Sequence Logits...")
    
    # BERT
    bert_fixed = np.zeros_like(recvae_logits)
    bert_fixed[u_target_idx] = bert_logits[u_source_row][:, b_reorder]
    
    # gSASRec
    gsas_fixed = np.zeros_like(recvae_logits)
    gsas_fixed[u_target_idx] = gsas_logits[u_source_row][:, g_reorder]

    # 합산
    print("⚖️ Normalizing and Summing...")
    norm_rvae = min_max_scale(recvae_logits)
    norm_m2vae = min_max_scale(m2vae_logits)
    norm_bert = min_max_scale(bert_fixed)
    norm_gsas = min_max_scale(gsas_fixed)
    norm_ease = min_max_scale(ease_logits)

    combined = (w_rvae * norm_rvae) + (w_m2vae * norm_m2vae) + (w_bert * norm_bert) + (w_gsas * norm_gsas) + (w_ease * norm_ease)

    # 진단 로그
    print("-" * 35)
    print(f"User 0 - RecVAE Top 10: {np.argsort(-norm_rvae[0])[:10]}")
    print(f"User 0 - M2VAE Top 10: {np.argsort(-norm_m2vae[0])[:10]}")
    print(f"User 0 - BERT Top 10: {np.argsort(-norm_bert[0])[:10]}")
    print(f"User 0 - gSAS Top 10: {np.argsort(-norm_gsas[0])[:10]}")
    print(f"User 0 - EASE Top 10: {np.argsort(-norm_ease[0])[:10]}")
    print("-" * 35)

    # 마스킹 및 저장
    train_rating_path = os.path.join(data_root, 'train/train_ratings.csv')
    train_df = pd.read_csv(train_rating_path)
    for u_id, i_id in zip(train_df['user'], train_df['item']):
        if u_id in s_u2idx and i_id in s_i2idx:
            combined[s_u2idx[u_id], s_i2idx[i_id]] = -1e9

    idx2item = m_static['idx2item']
    idx2user = {v: k for k, v in s_u2idx.items()}
    top_indices = np.argsort(-combined, axis=1)[:, :10]
    
    print("\n" + "="*40)
    print("👀 Example: Top 10 Recommendation & Scores for User 0")
    print("="*40)
    # 확인하고 싶은 유저의 인덱스 (예: 0번 유저)
    sample_uidx = 0
    
    u_real = idx2user[sample_uidx]
    print(f"User ID: {u_real}")
    
    for rank, i_idx in enumerate(top_indices[sample_uidx], 1):
        item_id = idx2item[i_idx]
        score = combined[sample_uidx, i_idx] # 최종 앙상블 점수
        
        # 각 모델별 Normalized Logit 값 추출
        val_rvae = norm_rvae[sample_uidx, i_idx]
        val_m2vae = norm_m2vae[sample_uidx, i_idx]
        val_bert = norm_bert[sample_uidx, i_idx]
        val_gsas = norm_gsas[sample_uidx, i_idx]
        val_ease = norm_ease[sample_uidx, i_idx]
        
        print(f"Rank {rank:2d} | Item: {item_id} | Final: {score:.6f}")
        print(f"       -> RecVAE: {val_rvae:.4f} | M2VAE: {val_m2vae:.4f} | BERT: {val_bert:.4f} | gSAS: {val_gsas:.4f} | EASE: {val_ease:.4f}")
    
    print("="*40 + "\n")
    
    final_preds = []
    for user_idx, item_indices in enumerate(top_indices):
        u_real = idx2user[user_idx]
        for i_idx in item_indices:
            final_preds.append([u_real, idx2item[i_idx]])
    
    # 저장 경로 설정
    submit_dir = os.path.join(ensemble_dir, "submit")
    if not os.path.exists(submit_dir):
        os.makedirs(submit_dir)
        
    output_filename = f'model_ensemble_(RecVAE({w_rvae})_M2VAE({w_m2vae})_BERT({w_bert})_gSAS({w_gsas})_EASE({w_ease})).csv'
    output_path = os.path.join(submit_dir, output_filename)
    
    pd.DataFrame(final_preds, columns=['user', 'item']).to_csv(output_path, index=False)
    print(f"✅ Done! File saved as {output_path}")

if __name__ == "__main__":
    ensemble(w_rvae=0.2, w_m2vae=0.2, w_bert=0.2, w_gsas=0.2, w_ease=0.2)