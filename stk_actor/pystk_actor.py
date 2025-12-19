from typing import List, Callable
from bbrl.agents import Agents, Agent
import gymnasium as gym

# Imports our Actor class
# IMPORTANT: note the relative import
from .actors import Actor, MyWrapper, ArgmaxActor, SamplingActor
from config import ppoconfig

#: The base environment name (you can change that)
env_name = "supertuxkart/full-v0"

#: Player name (you must change that)
player_name = "Example"

## Required parameters
seq_obs_keys = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")        
seq_raw_dims = {key:None for key in seq_obs_keys}
for key, space in env.observation_space.items():
    if key in seq_obs_keys:
        seq_raw_dims[key] = space.feature_space.shape
dfconfig = ppoconfig(device, seq_raw_dims).config

def get_wrappers() -> List[Callable[[gym.Env], gym.Wrapper]]:
    """Returns a list of additional wrappers to be applied to the base
    environment"""
    return [
        # Example of a custom wrapper
        lambda env: MyWrapper(env, option="1")
    ]


def get_actor(
    state: dict | None,
    observation_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
) -> Agent:
    """Creates a new actor (BBRL agent) that write into `action`

    :param state: The saved `stk_actor/pystk_actor.pth` (if it exists)
    :param observation_space: The environment observation space (with wrappers)
    :param action_space: The environment action space (with wrappers)
    :return: a BBRL agent
    """
    kwargs = {
        algo: 'PPO',
        config: dfconfig,
        device: device
    }
    actor = Actor(observation_space, action_space, **kwargs)

    # Returns a dummy actor
    if state is None:
        return SamplingActor(action_space)

    actor.load_state_dict(state)
    return Agents(actor, ArgmaxActor())
