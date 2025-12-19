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

    def __init__(self, algo, config, device):
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

    def forward(self, t: int, Batch):
        # Computes probabilities over actions
        fused_emb_value = self.value_obsEncoder(Batch)
        fused_emb_policy = self.policy_obsEncoder(Batch)
        value_estimate = self.valueHead(fused_emb_value)
        policy_output = self.policyHead(fused_emb_policy)
        
        return torch.exp(policy_output.log_porb)
    
    def get_value(self, t: int, Batch):
        fused_emb_value = self.value_obsEncoder(Batch)
        value_estimate = self.valueHead(fused_emb_value)
        return value_estimate
    
    def get_action_variables(self, t: int, Batch):
        fused_emb_policy = self.policy_obsEncoder(Batch)
        policy_output = self.policyHead(fused_emb_policy)
        return policy_output

class ArgmaxActor(Agent):
    """Actor that computes the action"""

    def forward(self, t: int):
        # Selects the best actions according to the policy
        raise NotImplementedError()


class SamplingActor(Agent):
    """Just sample random actions"""

    def __init__(self, action_space: gym.Space):
        super().__init__()
        self.action_space = action_space

    def forward(self, t: int):
        self.set(("action", t), torch.LongTensor([self.action_space.sample()]))
