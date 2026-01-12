import torch
import torch.nn as nn
import torch.nn.functional as F

class M2VAE(nn.Module):
    def __init__(self, input_dim, side_info_dims, weights=None, hidden_dim=600, latent_dim=200, dropout=0.5):
        super(M2VAE, self).__init__()
        
        # Side Info 처리를 위한 레이어
        self.side_project_dim = 64
        self.side_encoders = nn.ModuleDict({
            view: nn.Sequential(
                nn.Linear(dim, self.side_project_dim),
                nn.ReLU()
            ) for view, dim in side_info_dims.items()
        })
        
        # 통합 인코더의 입력 차원 계산
        self.total_input_dim = input_dim + (len(side_info_dims) * self.side_project_dim)
        
        # 통합 인코더 (Backbone)
        self.encoder_backbone = nn.Sequential(
            nn.Linear(self.total_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        
        self.mu_layer = nn.Linear(hidden_dim, latent_dim)
        self.var_layer = nn.Linear(hidden_dim, latent_dim)

        # 디코더
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.drop = nn.Dropout(dropout)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, x, side_info_dict):
        # Interaction 정규화
        h_inter = F.normalize(x, p=2, dim=1)
        
        # Side Info 프로젝션
        side_feats = []
        for view in self.side_encoders.keys():
            side_feats.append(self.side_encoders[view](side_info_dict[view]))
        
        # 모든 정보 결합 (Batch, Total_Input_Dim)
        combined_h = torch.cat([h_inter] + side_feats, dim=-1)
        combined_h = self.drop(combined_h)
        
        h = self.encoder_backbone(combined_h)
        mu = self.mu_layer(h)
        logvar = self.var_layer(h)

        # 샘플링 및 복원
        z = self.reparameterize(mu, logvar)
        recon_x = self.decoder(z)
        
        return recon_x, mu, logvar