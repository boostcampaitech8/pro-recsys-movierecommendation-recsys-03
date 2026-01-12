from scipy import sparse
import pandas as pd

def split_data(matrix):
    matrix = matrix.tocsr()
    train_rows, train_cols = [], []
    val_rows, val_cols = [], []
    
    for i in range(matrix.shape[0]):
        items = matrix.getrow(i).indices
        if len(items) > 1:
            val_rows.append(i)
            val_cols.append(items[-1])
            for item in items[:-1]:
                train_rows.append(i)
                train_cols.append(item)
        else:
            train_rows.append(i)
            train_cols.append(items[0])
            
    train_matrix = sparse.csr_matrix(([1]*len(train_rows), (train_rows, train_cols)), shape=matrix.shape)
    val_matrix = sparse.csr_matrix(([1]*len(val_rows), (val_rows, val_cols)), shape=matrix.shape)
    return train_matrix, val_matrix

def save_submission(pred_list, id2user, id2item, save_path='./output/submission.csv'):
    submission_data = []
    
    for user_idx, item_indices in enumerate(pred_list):
        original_user_id = id2user[user_idx]
        for item_id in item_indices:
            original_item_id = id2item[item_id]
            submission_data.append([original_user_id, original_item_id])
            
    df_sub = pd.DataFrame(submission_data, columns=['user', 'item'])
    df_sub.to_csv(save_path, index=False)
    print(f"Submission File Generated at {save_path}")


def save_candidates(results, save_path):
    df = pd.DataFrame(results, columns=['user', 'item', 'vae_score'])
    df.to_csv(save_path, index=False)
    print(f"Candidate file saved! Total rows: {len(df)} at {save_path}")