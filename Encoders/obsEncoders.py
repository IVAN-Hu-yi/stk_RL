import torch
import torch.nn as nn
import torch.nn.functional as F
from model_based_utils.mlpfac import MLP

box_obs = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']

discrete_action_keys = ['brake', 'drift', 'fire', 'nitro', 'rescue']
continuous_action_keys =  ['acceleration', 'steer']


class MLPboxObservationEncoder(nn.Module):
    
    """
        simple MLP encoder for discrete action spaces, ouput a latent vector for each action
    """
    def __init__(self, input_dim, config, output_dim, device):

        super().__init__()
        self.device = device
        self.encoder = MLP(input_dim, config, output_dim, device)
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)
        elif len(x.shape) == 2:
            x = x.unsqueeze(0)

        B, T, D = x.shape[0], x.shape[1], x.shape[2]
        x = x.view(B * T, D)
        z = self.encoder(x)
        return z.view(B, T, -1)

        
class seqObservationEncoder(nn.Module):
    """
        implements transformer-based encoder for sequence observation for later contecanation for one type
        Encodes sequence observation of shape:
        seq_obs: (batch_size, T, N, obs_dim) where T is the number of type steps, N is the number of entities
        seq_obs_mask: (batch_size, T, N) binary mask indicating valid entities
    """
    def __init__(
        self, 
        device,
        raw_dim,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        use_layer_norm: bool = True,
    ):
        super().__init__()
        self.device = device

        self.input_projection = nn.Linear(raw_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            device = device,
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='relu',
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            norm=nn.LayerNorm(d_model) if use_layer_norm else None,
        )
        self.attn_pool = nn.Linear(d_model, 1)
        self.d_model = d_model
        self.to(device)

    def forward(self, seq_obs, seq_obs_mask):
        seq_obs = seq_obs.to(self.device)
        seq_obs_mask = seq_obs_mask.to(self.device)

        if len(seq_obs.shape) == 3:
            seq_obs = seq_obs.unsqueeze(-1)
        if len(seq_obs_mask.shape) == 2:
            seq_obs_mask = seq_obs_mask.unsqueeze(-1)

        B, T, N, D = seq_obs.shape[0], seq_obs.shape[1], seq_obs.shape[2], seq_obs.shape[3]

        x = seq_obs.view(B * T, N, D)
        mask = ~seq_obs_mask.view(B * T, N).bool()

        x = self.input_projection(x) # B*T, N, d_model
        x = self.transformer(x, src_key_padding_mask=mask.to(self.device))

        attn_logits = self.attn_pool(x).squeeze(-1)
        attn_logits = attn_logits.masked_fill(mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)

        pooled = torch.sum(x * attn_weights, dim=1)

        embeddings = pooled.view(B, T, self.d_model)

        return embeddings

class obsEncoder(nn.Module):

    def __init__(self, config, device):
        super().__init__()

        self.device = device
        self.box_obs = ['attachment_time_left', 'aux_ticks', 'center_path', 'center_path_distance', 'distance_down_track', 'energy', 'front', 'max_steer_angle', 'shield_time', 'skeed_factor', 'velocity']

        self.seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
        box_output_dim = config['obsEncoder']['boxEncoder']['output_dim']
        self.boxEncoder = MLPboxObservationEncoder(17, config['obsEncoder']['boxEncoder'], box_output_dim, self.device)


        self.seqEncoder = nn.ModuleDict()
        feature_config = config['obsEncoder']['seqEncoder']
        d_model = feature_config['d_model']
        for obs_key in self.seq_obs:
            raw_dim = config['raw_dim'][obs_key][0] if len(config['raw_dim'][obs_key]) == 1 else 1 
            self.seqEncoder[obs_key] = seqObservationEncoder(
                self.device, raw_dim, **feature_config
            )
        self.output_dim = box_output_dim + d_model

    def forward(self, box_obs, seq_obs, seq_obs_mask):
        
        if isinstance(box_obs, list):
            box_obs = torch.cat(box_obs, dim=-1).to(self.device)

        box_embedding = self.boxEncoder(box_obs)
        seq_embeddings = {
            k: self.seqEncoder[k](v, seq_obs_mask[k])
            for k, v in seq_obs.items()
        }
        return torch.cat([box_embedding] + list(seq_embeddings.values()), dim=-1)


