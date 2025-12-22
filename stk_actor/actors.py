import gymnasium as gym
import torch

from .mlpValueNet import MLPValueModule
from .mlpPolicy import MLPPolicyModule
from .obsEncoders import obsEncoder
from .types import Transition, Batch
import copy

class MyWrapper(gym.ActionWrapper):
    def __init__(self, env, option: int):
        super().__init__(env)
        self.option = option

    def action(self, action):
        # We do nothing here
        return action


class Actor(torch.nn.Module):
    """Computes probabilities over action"""

    def __init__(self, observation_space, action_space, algo, config, device):
        super().__init__()

        self.device = device

        self.value_obsEncoder = obsEncoder(config, device)
        self.policy_obsEncoder = obsEncoder(config, device)

        self.policyHead = MLPPolicyModule(config, device)
        self.valueNet = MLPValueModule(config, device)

        # optional
        self.SRNet = None
        self.qNet = None

        self.algo = None

    def forward(self, t: int, obs):
        # encode observation
        # encode observation
        policy_latent = self.policy_obsEncoder(obs)

        # compute action distributions
        distDict = self.policyHead(policy_latent)

        self.set(("distribution", t), distDict)
        return distDict
    
class ArgmaxActor(torch.nn.Module):
    """Actor that computes the action"""

    def forward(self, t: int, distDict=None):
        # Selects the best actions according to the policy
        distDict = self.get(("distribution", t))
        actions = {}
        for key, dist in distDict.items():
            if isinstance(dist, torch.distributions.Bernoulli):
                actions[key] = (dist.logits >0).long()
            elif isinstance(dist, torch.distributions.Normal):
                actions[key] = dist.mean
                if key == 'steer':
                    actions[key] = torch.tanh(actions[key])
                else:
                    actions[key] = (torch.tanh(actions[key])+1)/2
        return self.set(("action", t), actions)

class SamplingActor(torch.nn.Module):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
