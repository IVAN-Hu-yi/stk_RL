import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent

from .mlpPolicy import MLPPolicyModule
from .obsEncoders import obsEncoder
from .obsWrapper import singleObsWrapper 

class MyWrapper(gym.ObservationWrapper): 
    def __init__(self, env, option: str):
        super().__init__(env)
        self.option = option

    def observation(self, obs):
        if isinstance(obs, dict):
            return {k: (np.array(v) if isinstance(v, (list, tuple)) else v) for k, v in obs.items()}
        return obs

class Actor(Agent):
    def __init__(self, observation_space, action_space, algo, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.policy_obsEncoder = obsEncoder(config, device)
        self.policyHead = MLPPolicyModule(config, device)
        self.Lmax = config.get('ReplayBuffer', {}).get('Lmax', 32)

    def test_inference(self):
        try:
            mock_box = torch.randn(1, 1, 17).to(self.device)
            mock_seq = {k: torch.randn(1, 1, self.Lmax, self.config['raw_dim'][k][0]).to(self.device) for k in self.policy_obsEncoder.seq_keys}
            mock_mask = {k: torch.ones(1, 1, self.Lmax, dtype=torch.bool).to(self.device) for k in self.policy_obsEncoder.seq_keys}
            with torch.no_grad():
                latent = self.policy_obsEncoder(mock_box, mock_seq, mock_mask)
                self.policyHead(latent)
            return True
        except:
            return False

    def _get_obs_dict(self, t):
        obs_keys = ['velocity', 'items_position', 'items_type', 'karts_position', 
                    'paths_distance', 'paths_width', 'paths_start', 'paths_end',
                    'attachment_time_left', 'aux_ticks', 'center_path', 
                    'center_path_distance', 'distance_down_track', 'energy', 
                    'front', 'max_steer_angle', 'shield_time', 'skeed_factor']
        obs_dict = {}
        for k in obs_keys:
            val = self.get((f"env/env_obs/{k}", t))
            if val is not None:
                if torch.is_tensor(val): val = val.detach().cpu().numpy()
                if isinstance(val, np.ndarray) and val.ndim > 0 and val.shape[0] == 1: val = val[0]
                obs_dict[k] = val
        return obs_dict

    def forward(self, t: int, **kwargs):
        # 1. 快速获取数据
        raw_obs = self._get_obs_dict(t)
        if not raw_obs: return
        
        # 2. 预处理 (在 CPU 上完成以节省显存/调度开销)
        box_raw, seq_raw, mask_raw = singleObsWrapper(raw_obs, Lmax=self.Lmax)
        
        # 3. 极简 Tensor 构造
        box_t = torch.as_tensor(np.concatenate([np.atleast_1d(x) for x in box_raw]), dtype=torch.float32, device=self.device).view(1, 1, -1)
        seq_t = {k: torch.as_tensor(np.array(v), dtype=torch.float32, device=self.device).view(1, 1, self.Lmax, -1) for k, v in seq_raw.items()}
        mask_t = {k: torch.as_tensor(np.array(v), dtype=torch.bool, device=self.device).view(1, 1, self.Lmax) for k, v in mask_raw.items()}
        
        # 4. 严格执行推理
        with torch.no_grad():
            latent = self.policy_obsEncoder(box_t, seq_t, mask_t)
            out = self.policyHead(latent)
            
            distributions = out.distDict if hasattr(out, 'distDict') else out
            if not isinstance(distributions, dict): distributions = distributions.__dict__

            # 默认值覆盖法
            actions = {
                'acceleration': torch.tensor([[1.0]], device=self.device),
                'brake': torch.tensor([[0.0]], device=self.device),
                'steer': torch.tensor([[0.0]], device=self.device),
                'drift': torch.zeros((1, 1), dtype=torch.long, device=self.device),
                'nitro': torch.zeros((1, 1), dtype=torch.long, device=self.device),
                'rescue': torch.zeros((1, 1), dtype=torch.long, device=self.device),
                'fire': torch.zeros((1, 1), dtype=torch.long, device=self.device)
            }

            for key, dist in distributions.items():
                if not isinstance(dist, torch.distributions.Distribution): continue
                if isinstance(dist, torch.distributions.Bernoulli):
                    actions[key] = (dist.logits > 0).long().view(1, -1)
                else:
                    val = torch.tanh(dist.mean) if key == 'steer' else (torch.tanh(dist.mean) + 1) / 2
                    actions[key] = val.view(1, -1)

            # 统一写回
            for key, val in actions.items():
                self.set((f"action/{key}", t), val)