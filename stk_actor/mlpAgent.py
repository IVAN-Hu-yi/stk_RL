from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from dataclasses import asdict, is_dataclass
from collections import defaultdict
from typing import Dict, Any, Optional, Literal

from .basePolicyModule import PolicyModule
from .mlpValueNet import ValueModule, QModule, SRModule
from .basebuffer import ReplayBuffer
from .types import Transition, Batch
from .base import RLAlgorithm
import copy

class baseAgent(nn.Module):
    """
     wrapper for interacting with agents'modules (policy, valueNet, qNet, SRNet) and RL algrithms
    """
    def __init__(
        self,
        obsEncoder =None,
        policy =None,
        valueNet = None,
        qNet = None,
        SRNet: =None,
        mcReturn = None,
    ):

        self.policy  = policy
        self.ValueNet = ValueNet
        self.QNet   = qNet
        self.SRNet   = SRNet
        self.policy_obsEncoder = copy.deepcopy(obsEncoder)
        self.value_obsEncoder = copy.deepcopy(obsEncoder)
        self.SR_obsEncoder = copy.deepcopy(obsEncoder)
        self.Q_obsEncoder = copy.deepcopy(obsEncoder)
        self.mcReturn = mcReturn


class MLPAgent(baseAgent):
    def __init__(
        algo: str,
        device,
        name,
        obsEncoder =None,
        policy =None,
        valueNet = None,
        qNet = None,
        SRNet: =None,
        config,
    ):
        super().__init__(obsEncoder, policy, valueNet, qNet, SRNet)
        self.name = name
        self.device = device

        # define algorithm
        AlgorithmClass = get_algorithm(algo)
        self.algo = AlgorithmClass(self, config['algo_config'])

