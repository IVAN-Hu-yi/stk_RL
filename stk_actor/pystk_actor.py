from typing import List, Callable
import gymnasium as gym
import torch
from pathlib import Path
from .actors import Actor, MyWrapper
from .config import ppoconfig

env_name = "supertuxkart/full-v0"
player_name = "XHB_Kart"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    return [lambda env: MyWrapper(env, option="1")]

def get_actor(state, observation_space, action_space):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_dim = {
        'velocity': (3,), 'items_position': (3,), 'items_type': (1,),
        'karts_position': (3,), 'paths_distance': (2,), 'paths_width': (1,),
        'paths_start': (3,), 'paths_end': (3,), 'attachment_time_left': (1,),
        'aux_ticks': (1,), 'center_path': (3,), 'center_path_distance': (1,),
        'distance_down_track': (1,), 'energy': (1,), 'front': (3,),
        'max_steer_angle': (1,), 'shield_time': (1,), 'skeed_factor': (1,)
    }

    cfg_obj = ppoconfig(device, raw_dim)
    actor = Actor(observation_space, action_space, "PPO", cfg_obj.config, device)

    model_path = Path(__file__).parent / "pystk_actor.pth"
    if state is None and model_path.exists():
        ckpt = torch.load(model_path, map_location=device)
        state = ckpt.get("actor_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

    if state:
        actor.load_state_dict(state, strict=False)
        print("Model state loaded.")
    
    # 自动测试模型推理链路
    actor.test_inference()
    
    actor.eval()
    return actor