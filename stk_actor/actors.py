import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from bbrl.agents import Agent

# 导入项目模块
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
        """测试函数：确保模型在单步推理时不会崩溃"""
        print("--- Running Actor Internal Test ---")
        try:
            mock_box = torch.randn(1, 1, 17).to(self.device)
            mock_seq = {k: torch.randn(1, 1, self.Lmax, self.config['raw_dim'][k][0]).to(self.device) for k in self.policy_obsEncoder.seq_keys}
            mock_mask = {k: torch.ones(1, 1, self.Lmax, dtype=torch.bool).to(self.device) for k in self.policy_obsEncoder.seq_keys}
            
            with torch.no_grad():
                latent = self.policy_obsEncoder(mock_box, mock_seq, mock_mask)
                self.policyHead(latent)
            print("--- Internal Test Passed! ---")
            return True
        except Exception as e:
            print(f"--- Internal Test Failed: {e} ---")
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
                if isinstance(val, np.ndarray) and len(val.shape) > 0 and val.shape[0] == 1: val = val[0]
                obs_dict[k] = val
        return obs_dict

    def forward(self, t: int, **kwargs):
        raw_obs = self._get_obs_dict(t)
        if not raw_obs: return
        
        box_raw, seq_raw, mask_raw = singleObsWrapper(raw_obs, Lmax=self.Lmax)
        
        # 1. 准备输入数据
        box_np = np.concatenate([np.atleast_1d(x.cpu() if torch.is_tensor(x) else x) for x in box_raw])
        box_t = torch.as_tensor(box_np, dtype=torch.float32).to(self.device).view(1, 1, -1)
        seq_t = {k: torch.as_tensor(v, dtype=torch.float32).to(self.device).view(1, 1, self.Lmax, -1) for k, v in seq_raw.items()}
        mask_t = {k: torch.as_tensor(v, dtype=torch.bool).to(self.device).view(1, 1, self.Lmax) for k, v in mask_raw.items()}
        
        # 2. 推理
        with torch.no_grad():
            latent = self.policy_obsEncoder(box_t, seq_t, mask_t)
            out = self.policyHead(latent)

        # 3. 提取分布
        distributions = out.distDict if hasattr(out, 'distDict') else out
        if not isinstance(distributions, dict): distributions = distributions.__dict__

        # 4. 【核心修复】初始化 PySTK2 必需的所有动作键
        # 即使模型没预测某个动作，我们也得给环境一个默认 Tensor 占位
        final_actions = {
            'acceleration': torch.tensor([1.0]), # 默认全速前进
            'brake': torch.tensor([0.0]),        # 默认不刹车
            'steer': torch.tensor([0.0]),        # 默认直行
            'drift': torch.tensor([0]),          # 默认不漂移 (Long)
            'nitro': torch.tensor([0]),          # 默认不喷射 (Long)
            'rescue': torch.tensor([0]),
            'fire': torch.tensor([0])
        }

        # 5. 用模型的预测结果覆盖默认值
        for key, dist in distributions.items():
            if not isinstance(dist, torch.distributions.Distribution): continue
            
            if isinstance(dist, torch.distributions.Bernoulli):
                # 离散动作处理 (drift, nitro, fire 等)
                val = (dist.logits > 0).long()
            else:
                # 连续动作处理 (steer, acceleration, brake)
                mu = dist.mean
                if key == 'steer':
                    val = torch.tanh(mu)
                else:
                    val = (torch.tanh(mu) + 1) / 2
            
            # 将模型输出转换回 Tensor 并更新字典
            final_actions[key] = val.detach().cpu().view(-1)

        # 6. 将最终补全的动作字典写回 Workspace
        for key, val in final_actions.items():
            # 框架期望 [Batch, Dim]
            self.set((f"action/{key}", t), val.to(self.device).view(1, -1))