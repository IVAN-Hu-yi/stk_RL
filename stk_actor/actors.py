import gymnasium as gym
import torch
from bbrl.agents import Agent
from ValueModule.mlpValueNet import MLPValueModule
from PolicyModule.mlpPolicy import MLPPolicyModule
from Encoders.obsEncoders import obsEncoder
from buffers.types import Transition, Batch
import copy
from config import ppoconfig
from algorithm.base import register_algorithm, get_algorithm


class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(Agent):
    """Computes probabilities over action"""

    def __init__(self, observation_space, action_space, algo, config, device):
        super().__init__()

        self.policyHead = MLPPolicyModule(config, device)
        self.valueNet = MLPValueModule(config, device)
        self.value_obsEncoder = obsEncoder(config, device)
        self.policy_obsEncoder = obsEncoder(config, device)

        # optional
        self.SRNet = None
        self.qNet = None

        AlgorithmClass = get_algorithm(algo)
        self.algo = AlgorithmClass(agent=self, config=config)

    def forward(self, t: int, obs):
        self.set(("obs", t), obs)
        outputs = self.algo.select_action(obs, eval_mode=True)
        distribution = outputs[2]
        self.set(("action", t), outputs[0])
        self.set(("distribution", t), distribution)
        return distribution
    
class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int, distDict=None):
        # Selects the best actions according to the policy
        distDict = self.get(("distribution", t))
        actions = {}
        for key, dist in distDict.items():
            elif isinstance(dist, torch.distributions.Bernoulli):
                actions[key] = (dist.logits >0).long()
            elif isinstance(dist, torch.distributions.Normal):
                actions[key] = dist.mean
                if key == 'steer':
                    actions[key] = torch.tanh(actions[key])
                else:
                    actions[key] = (torch.tanh(actions[key])+1)/2
        return self.set(("action", t), actions)

class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
