from .basePolicyModule import PolicyModule, PolicyOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from model_based_utils.mlpfac import MLP

LOG2 = torch.log(torch.tensor(2.0))
def cont_log_prob(action, mean, logstd, device, affine_scale=False):
    dist = D.Normal(mean, torch.exp(logstd))  
    logp = dist.log_prob(action).sum(dim=-1)
    correction = (2.0 * (action + F.softplus(-2.0 * action) - LOG2)).sum(dim=-1)
    logp -= correction
    if affine_scale:
        logp -= LOG2 
    return logp 

class LogStd(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(dim))   

class MLPPolicyModule(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config
        self.discrete_action_keys = ['brake', 'drift', 'fire', 'nitro', 'rescue']
        self.continuous_action_keys =  ['acceleration', 'steer']

        self.action_heads = nn.ModuleDict()
        self.logstds = nn.ParameterDict()

        seq_obs = ['items_position', 'items_type', 'karts_position', 'paths_distance', 'paths_end', 'paths_start', 'paths_width']

        input_dim = self.config["obsEncoder"]["boxEncoder"]["output_dim"] + len(seq_obs)*self.config["obsEncoder"]["seqEncoder"]["d_model"]

        for key in self.discrete_action_keys:
            self.action_heads[key] = MLP(input_dim, self.config["policy_config"]["disc_action"], 1, device)

        # action heads predict mean and logstd for continuous actions
        for key in self.continuous_action_keys:
            self.action_heads[key]= MLP(input_dim, self.config["policy_config"]["cont_action"][key], 1, device)
            self.logstds[key] = LogStd(1)

        self.optimizer = torch.optim.Adam(self.action_heads.parameters(), lr=config["policy_config"]['lr'])

        self.to(device)

    def forward(self, fused_emb, hidden=None, requires_grad=True) -> PolicyOutput:
        
        """
            inputs:
                fused_emb: fused embedding from observation encoder
        """
        actions = {}
        entropies = {}
        log_prob = {}
        distributions = {}
        for key in self.discrete_action_keys :
            action_logits = self.action_heads[key](fused_emb).squeeze(-1)
            out = self.action_heads[key](fused_emb)
            action_logits = torch.clamp(action_logits, -20.0, 20.0)
            dist = D.Bernoulli(logits=action_logits)
            actions[key] = dist.sample()
            entropies[key] = dist.entropy()
            log_prob[key] = dist.log_prob(actions[key])
            distributions[key] = dist
        for key in self.continuous_action_keys:
            action_mean = self.action_heads[key](fused_emb)
            action_logstd = self.logstds[key].log_std.view(1, 1, 1).expand_as(action_mean)
            dist = D.Normal(action_mean, torch.exp(action_logstd))
            entropies[key] = dist.entropy().sum(dim=-1)
            action = dist.sample()
            distributions[key] = dist
            if key == 'steer':
                actions[key] = torch.tanh(action)
                log_prob[key] = cont_log_prob(action, action_mean, action_logstd, self.device).contiguous()
            else:
                actions[key] = (torch.tanh(action)+1)/2 # map to [0,1]
                log_prob[key] = cont_log_prob(action, action_mean, action_logstd, self.device, True).contiguous()
        return PolicyOutput(
            action=actions,
            log_prob=log_prob,
            entropy=entropies,
            new_hidden=None,
            extra={"distribution": distributions}
        )    
