import torch
import torch.nn as nn
from .model_based_utils.mlpfac import MLP
import torch.optim

class MLPValueModule(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
        input_dim = self.config["obsEncoder"]["boxEncoder"]["output_dim"] + len(seq_obs)* self.config["obsEncoder"]["seqEncoder"]["d_model"]
        self.values = MLP(input_dim, config["value_config"], 1, device)
        self.optimizer = torch.optim.Adam(self.values.parameters(), lr=config["value_config"]['lr'])

    def forward(self, fused_obs_emb):
        return self.values(fused_obs_emb.to(self.device)).squeeze(-1)
