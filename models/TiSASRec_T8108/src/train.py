"""
TiSASRec 학습 스크립트

사용법:
    python train.py
    python train.py --epochs 100 --lr 0.0001
"""

import argparse
import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import torch
from logging import getLogger
from tqdm import tqdm

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger, set_color

from models.tisasrec import TiSASRec


class SimpleTrainer:
    """TiSASRec 학습용 Trainer"""
    
    def __init__(self, config, model):
        self.config = config
        self.model = model
        self.logger = getLogger()
        
        self.device = config['device']
        self.epochs = config['epochs']
        self.eval_step = config['eval_step'] if 'eval_step' in config.final_config_dict else 1
        self.stopping_step = config['stopping_step'] if 'stopping_step' in config.final_config_dict else 10
        self.checkpoint_dir = config['checkpoint_dir']
        
        self.learning_rate = config['learning_rate']
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=config['weight_decay'] if 'weight_decay' in config.final_config_dict else 0.0
        )
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_valid_score = -float('inf')
        self.best_valid_result = None
        self.cur_step = 0
    
    def _train_epoch(self, train_data, epoch_idx):
        self.model.train()
        total_loss = 0.0
        
        progress = tqdm(train_data, desc=f"Train Epoch {epoch_idx+1}")
        for batch_idx, interaction in enumerate(progress):
            interaction = interaction.to(self.device)
            self.optimizer.zero_grad()
            
            loss = self.model.calculate_loss(interaction)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())
        
        return total_loss / len(train_data)
    
    @torch.no_grad()
    def _valid_epoch(self, valid_data):
        self.model.eval()
        
        total_recall = 0.0
        total_users = 0
        
        for batched_data in tqdm(valid_data, desc="Validating"):
            interaction, history_index, positive_u, positive_i = batched_data
            interaction = interaction.to(self.device)
            
            scores = self.model.full_sort_predict(interaction)
            scores = scores.view(-1, self.model.n_items)
            
            # History masking
            scores[history_index] = -float('inf')
            
            # Top-10 추천
            _, topk_idx = torch.topk(scores, k=10, dim=1)
            
            # Recall@10 계산
            for i in range(len(positive_u)):
                pos_item = positive_i[i]
                if isinstance(pos_item, torch.Tensor):
                    pos_item = pos_item.item() if pos_item.dim() == 0 else pos_item.cpu().numpy().tolist()
                
                if isinstance(pos_item, (list, tuple)):
                    pos_set = set(pos_item)
                else:
                    pos_set = {int(pos_item)}
                
                pred_set = set(topk_idx[i].cpu().numpy().tolist())
                hits = len(pos_set & pred_set)
                total_recall += hits / len(pos_set) if len(pos_set) > 0 else 0
                total_users += 1
        
        recall_at_10 = total_recall / total_users if total_users > 0 else 0
        return {'Recall@10': recall_at_10}
    
    def fit(self, train_data, valid_data):
        for epoch_idx in range(self.epochs):
            train_loss = self._train_epoch(train_data, epoch_idx)
            self.logger.info(f"Epoch {epoch_idx+1}/{self.epochs} - Train Loss: {train_loss:.4f}")
            
            if (epoch_idx + 1) % self.eval_step == 0:
                valid_result = self._valid_epoch(valid_data)
                valid_score = valid_result['Recall@10']
                self.logger.info(f"Valid Recall@10: {valid_score:.4f}")
                
                if valid_score > self.best_valid_score:
                    self.best_valid_score = valid_score
                    self.best_valid_result = valid_result
                    self.cur_step = 0
                    self._save_checkpoint(epoch_idx)
                    self.logger.info(f"*** New best model! Recall@10: {valid_score:.4f} ***")
                else:
                    self.cur_step += 1
                    if self.cur_step >= self.stopping_step:
                        self.logger.info(f"Early stopping at epoch {epoch_idx+1}")
                        break
        
        return self.best_valid_score, self.best_valid_result
    
    def _save_checkpoint(self, epoch_idx):
        path = os.path.join(self.checkpoint_dir, 'TiSASRec-best.pth')
        torch.save({
            'epoch': epoch_idx,
            'state_dict': self.model.state_dict(),
            'best_valid_score': self.best_valid_score,
        }, path)
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['state_dict'])
    
    @torch.no_grad()
    def test(self, test_data):
        return self._valid_epoch(test_data)


def train(**kwargs):
    # 기본 설정
    config_dict = {
        'data_path': os.path.join(SCRIPT_DIR, '..', 'dataset'),
        'checkpoint_dir': os.path.join(SCRIPT_DIR, '..', 'output', 'checkpoints'),
        'USER_ID_FIELD': 'user_id',
        'ITEM_ID_FIELD': 'item_id',
        'TIME_FIELD': 'timestamp',
        'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
        
        # 학습
        'epochs': 200,
        'train_batch_size': 256,
        'eval_batch_size': 256,
        'learning_rate': 0.001,
        
        # 평가 - full sort 모드
        'eval_args': {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'group_by': 'user',
            'order': 'TO',
            'mode': 'full'
        },
        'metrics': ['Recall', 'NDCG'],
        'topk': [10],
        'valid_metric': 'Recall@10',
        
        # Early stopping
        'stopping_step': 10,
        'eval_step': 1,
        
        # 모델 (SASRec 기본값 사용)
        'hidden_size': 64,
        'inner_size': 256,
        'n_layers': 2,
        'n_heads': 2,
        'hidden_dropout_prob': 0.5,
        'attn_dropout_prob': 0.5,
        'hidden_act': 'gelu',
        'layer_norm_eps': 1e-12,
        'initializer_range': 0.02,
        'MAX_ITEM_LIST_LENGTH': 50,
        'loss_type': 'CE',
        
        # TiSASRec 전용
        'time_span': 256,  # deprecated, 호환성 위해 유지
        'max_time_interval': 256,  # 논문의 k값 (최대 시간 간격 클리핑)
        
        # CE loss 사용 시 negative sampling 비활성화 (필수)
        'train_neg_sample_args': None,
    }
    
    config_dict.update(kwargs)
    
    # Config 생성
    config = Config(
        model='SASRec',
        dataset=config_dict.get('dataset', 'ml_movie'),
        config_dict=config_dict
    )
    
    # 초기화
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    
    logger.info("=" * 60)
    logger.info("TiSASRec Training")
    logger.info("=" * 60)
    logger.info(f"Dataset: {config['dataset']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Epochs: {config['epochs']}")
    logger.info(f"Hidden Size: {config['hidden_size']}")
    logger.info(f"Time Span: {config['time_span']}")
    logger.info(f"Max Time Interval (k): {config['max_time_interval']}")
    logger.info("=" * 60)
    
    # 데이터셋
    logger.info(set_color('Creating dataset...', 'yellow'))
    dataset = create_dataset(config)
    logger.info(dataset)
    
    # 데이터 준비
    logger.info(set_color('Preparing data...', 'yellow'))
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 모델
    logger.info(set_color('Building model...', 'yellow'))
    model = TiSASRec(config, train_data.dataset).to(config['device'])
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total Parameters: {total_params:,}")
    
    # 학습
    trainer = SimpleTrainer(config, model)
    
    logger.info(set_color('Starting training...', 'yellow'))
    start_time = datetime.now()
    
    best_score, best_result = trainer.fit(train_data, valid_data)
    
    training_time = datetime.now() - start_time
    logger.info(f"\nTraining completed in {training_time}")
    logger.info(f"Best Recall@10: {best_score:.4f}")
    
    # 테스트
    logger.info(set_color('\nTesting...', 'yellow'))
    checkpoint_path = os.path.join(config['checkpoint_dir'], 'TiSASRec-best.pth')
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    
    test_result = trainer.test(test_data)
    logger.info(f"Test Recall@10: {test_result['Recall@10']:.4f}")
    
    # 결과 저장
    result_file = os.path.join(config['checkpoint_dir'], 'result.txt')
    with open(result_file, 'w') as f:
        f.write(f"Training time: {training_time}\n")
        f.write(f"Best Valid Recall@10: {best_score:.4f}\n")
        f.write(f"Test Recall@10: {test_result['Recall@10']:.4f}\n")
    
    return best_score, best_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml_movie')
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden_size', type=int, default=None)
    parser.add_argument('--inner_size', type=int, default=None, help='FFN inner dimension (default: hidden_size * 4)')
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--max_seq_length', type=int, default=None)
    parser.add_argument('--time_span', type=int, default=None, help='(deprecated) Use --max_time_interval')
    parser.add_argument('--max_time_interval', type=int, default=None, help='Max time interval clipping (k in paper)')
    parser.add_argument('--dropout', type=float, default=None, help='Hidden & attention dropout (0.0-1.0)')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    kwargs = {'dataset': args.dataset, 'gpu_id': args.gpu_id, 'seed': args.seed}
    
    if args.epochs: kwargs['epochs'] = args.epochs
    if args.batch_size:
        kwargs['train_batch_size'] = args.batch_size
        kwargs['eval_batch_size'] = args.batch_size
    if args.lr: kwargs['learning_rate'] = args.lr
    if args.hidden_size: kwargs['hidden_size'] = args.hidden_size
    if args.inner_size: kwargs['inner_size'] = args.inner_size
    if args.n_layers: kwargs['n_layers'] = args.n_layers
    if args.n_heads: kwargs['n_heads'] = args.n_heads
    if args.max_seq_length: kwargs['MAX_ITEM_LIST_LENGTH'] = args.max_seq_length
    if args.time_span: kwargs['max_time_interval'] = args.time_span  # 호환성
    if args.max_time_interval: kwargs['max_time_interval'] = args.max_time_interval
    if args.dropout is not None:
        kwargs['hidden_dropout_prob'] = args.dropout
        kwargs['attn_dropout_prob'] = args.dropout
    
    train(**kwargs)


if __name__ == "__main__":
    main()