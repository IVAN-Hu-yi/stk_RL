from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym
# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor
from .config import ppoconfig
import torch
from .actors import Actor

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

    raw_dim = {}

    for key, space in observation_space.items():
        # Box scalar or vector
        if hasattr(space, "shape") and space.shape is not None:
            if len(space.shape) == 0:
                raw_dim[key] = (1,)
            else:
                raw_dim[key] = (space.shape[0],)
        else:
            raw_dim[key] = (1,)

    # === 1. 构造一个最小可用 config dict ===
    config = {
        "raw_dim": raw_dim,
        
        "obsEncoder": {
            "boxEncoder": {
                "hidden_size": [128, 128],
                "hidden_activation": ["relu", "relu"],
                "output_activation": [""],
                "output_dim": 128,
            },
            "seqEncoder": {
                "d_model": 64,
                "n_heads": 4,
                "n_layers": 1,
                "dropout": 0.0,
            },
        },

        # ⭐ 关键：policy_config
        "policy_config": {
            "lr": 1e-4,
            "disc_action": {
                "hidden_size": [256, 256],
                "hidden_activation": ["relu", "relu"],
                "output_activation": [""],
            },
            "cont_action": {
                "steer": {
                    "hidden_size": [256, 256],
                    "hidden_activation": ["relu", "relu"],
                    "output_activation": [""],
                },
                "acceleration": {
                    "hidden_size": [256, 256],
                    "hidden_activation": ["relu", "relu"],
                    "output_activation": [""],
                },
            },
        },

        # ⭐ value 网络用
        "value_config": {
            "lr": 1e-4,
            "hidden_size": [256, 256],
            "hidden_activation": ["relu", "relu"],
            "output_activation": [""],
        },
    }



    actor = Actor(
        observation_space=observation_space,
        action_space=action_space,
        algo="None",
        config=config,
        device=device,
    )

    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())


