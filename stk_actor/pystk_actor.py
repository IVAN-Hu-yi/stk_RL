from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor
from .config import ppoconfig
import torch

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0"

#: Player name (you must change that)
player_name = "Example"

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: MyWrapper(env, option="1")
    ]

def get_actor(state, observation_space, action_space):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === 1. 构造一个最小可用 config dict ===
    config = {
        "obsEncoder": {
            "boxEncoder": {
                "output_dim": 128,   # ⚠️ 和你训练时一致即可
            },
            "seqEncoder": {
                "d_model": 64,
            },
        },
        "policy": {
            "hidden_sizes": [256, 256],
        },
        "value": {
            "hidden_sizes": [256, 256],
        },
    }

    from .actors import Actor

    actor = Actor(
        observation_space=observation_space,
        action_space=action_space,
        algo="PPO",
        config=config,
        device=device,
    )

    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


