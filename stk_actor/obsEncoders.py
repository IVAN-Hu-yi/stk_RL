import torch
import torch.nn as nn
from .mlpfac import MLP

class MLPboxObservationEncoder(nn.Module):
    def __init__(self, input_dim, config, output_dim, device):
        super().__init__()
        self.encoder = MLP(input_dim, config, output_dim, device)
        self.to(device)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B * T, D)
        z = self.encoder(x)
        return z.view(B, T, -1)

class seqObservationEncoder(nn.Module):
    def __init__(self, device, input_dim, d_model, n_heads, n_layers, dropout, **kwargs):
        super().__init__()
        self.device = device
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # ⚠️ 关键修改：增加 enable_nested_tensor=False
        # 这会防止某些 PyTorch 版本在处理 mask 时卡死
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=n_layers,
            enable_nested_tensor=False 
        )
        self.to(device)

    def forward(self, x, seq_obs_mask):
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)
        x = self.input_proj(x)
        
        # 将 mask 转换为 float 以提高不同 PyTorch 版本的计算兼容性
        mask = ~seq_obs_mask.view(B * T, N).bool()
        out = self.transformer(x, src_key_padding_mask=mask)
        out = out.mean(dim=1) 
        return out.view(B, T, -1)

class obsEncoder(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.seq_keys = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
        
        box_out_dim = config['obsEncoder']['boxEncoder']['output_dim']
        self.boxEncoder = MLPboxObservationEncoder(17, config['obsEncoder']['boxEncoder'], box_out_dim, device)
        
        self.seqEncoder = nn.ModuleDict()
        seq_cfg = config['obsEncoder']['seqEncoder']
        for k in self.seq_keys:
            d_in = config['raw_dim'][k][0]
            self.seqEncoder[k] = seqObservationEncoder(device, d_in, **seq_cfg)
            
        self.output_dim = box_out_dim + len(self.seq_keys) * seq_cfg['d_model']

    def forward(self, box_obs, seq_obs, seq_obs_mask):
        b_emb = self.boxEncoder(box_obs) 
        s_embs = [self.seqEncoder[k](seq_obs[k], seq_obs_mask[k]) for k in self.seq_keys]
        return torch.cat([b_emb] + s_embs, dim=-1)