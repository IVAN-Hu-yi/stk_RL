import torch
import torch.nn.functional as F
from .base import RLAlgorithm, register_algorithm
from .buffers.types import Batch
from typing import Dict, Any, Tuple
from .wrappers.obsWrapper import singleObsWrapper

@register_algorithm('PPO')
class PPO(RLAlgorithm):

    def __init__(
        self,
        agent,
        config: Dict[str, Any],
    ):
        self.cfg = config 
        super().__init__(agent, self.cfg['algo_config'])

    def select_action(
        self,
        obs,
        hidden=None,
        eval_mode=False
    ) -> Tuple[Any, Any, Any, Any]:
        """
        Select action using the policy module.
        """
        transformedObs = singleObsWrapper(obs, Lmax=self.cfg['ReplayBuffer']['Lmax'])
        fused_emb_policy = self.agent.policy_obsEncoder(*transformedObs)

        with torch.no_grad():
            policy_output = self.agent.policyHead(fused_emb_policy, hidden=hidden)
        if not eval_mode:
            return (
                policy_output.action,
                policy_output.log_prob,
                policy_output.new_hidden
            )
        else:
            return (
                policy_output.action,
                policy_output.new_hidden,
                policy_output.extra["distribution"]
            )

    def _compute_gae(self, deltas, dones, gamma, lam):
        """
        deltas: [B, T]
        dones:  [B, T]
        returns:
            advantages: [B, T]
        """
        B, T = deltas.shape
        advantages = torch.zeros_like(deltas)

        gae = torch.zeros(B, device=deltas.device)

        for t in reversed(range(T)):
            mask = ~dones[:, t]
            gae = deltas[:, t] + gamma * lam * mask * gae
            advantages[:, t] = gae

        return advantages

    def update(
        self,
        batch: Batch
    ) -> Dict[str, float]:
        """
        Update the policy and value networks using PPO.
        """
        total_policy_loss = 0.0
        total_value_loss = 0.0

        if self.agent.valueNet is not None:
            fused_obs_emb_value = self.agent.value_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
            next_fused_obs_emb_value = self.agent.value_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)

        fused_obs_emb_policy = self.agent.policy_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
        next_fused_obs_emb_policy = self.agent.policy_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)

        if self.agent.qNet is not None:
            fused_obs_emb_Q = self.agent.Q_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
            next_fused_obs_emb_Q = self.agent.Q_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)
            
        if self.agent.SRNet is not None:
            fused_obs_emb_SR = self.agent.SR_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
            next_fused_obs_emb_SR = self.agent.SR_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)
        
        # Compute advantages and returns
        with torch.no_grad():
            old_values = self.agent.valueNet(fused_obs_emb_value)
            old_next_values = self.agent.valueNet(next_fused_obs_emb_value)
            dones = (batch.dones)
            deltas = batch.rewards + self.config['gamma'] * old_next_values * (~dones) - old_values
            advantages = self._compute_gae(deltas, dones, self.config['gamma'], self.config['lambda'])
            returns = advantages + old_values

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.config['ppo_epochs']):

        ## Updated embeddings for each epoch
            if self.agent.valueNet is not None:
                fused_obs_emb_value = self.agent.value_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
                next_fused_obs_emb_value = self.agent.value_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)

                fused_obs_emb_policy = self.agent.policy_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
                next_fused_obs_emb_policy = self.agent.policy_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)

            if self.agent.qNet is not None:
                fused_obs_emb_Q = self.agent.Q_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
                next_fused_obs_emb_Q = self.agent.Q_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)
                
            if self.agent.SRNet is not None:
                fused_obs_emb_SR = self.agent.SR_obsEncoder(batch.box_obs, batch.seq_obs, batch.seq_mask)
                next_fused_obs_emb_SR = self.agent.SR_obsEncoder(batch.next_box_obs, batch.next_seq_obs, batch.next_seq_mask)

            # Get current policy outputs
            policy_output = self.agent.policyHead(fused_obs_emb_policy)
            log_probs = policy_output.log_prob
            entropies = policy_output.entropy

            # Compute ratios
            log_prob_total = torch.stack(list(log_probs.values()), dim=0).sum(dim=0)
            batch_log_prob_total = torch.stack(list(batch.log_prob.values()), dim=0).sum(dim=0)
            ratios = torch.exp(log_prob_total - batch_log_prob_total)

            # Compute surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1.0 - self.config['clip_eps'], 1.0 + self.config['clip_eps']) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy_loss = torch.stack(list(entropies.values()), dim=0).sum(dim=0)
            entropy_loss = entropy_loss.mean()
    
            
            policy_loss -= self.config['entropy_coef'] * entropy_loss.item()

            # Value function loss
            value_estimates = self.agent.valueNet(fused_obs_emb_value)
            value_loss = F.mse_loss(value_estimates, returns)

            # Backpropagation
            assert torch.isfinite(policy_loss).all(), f"loss is not finite: {policy_loss.item()}"
            self.agent.policyHead.optimizer.zero_grad()
            self.agent.valueNet.optimizer.zero_grad()
            policy_loss.backward()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.agent.policyHead.parameters(),
                max_norm=0.5
            )
            torch.nn.utils.clip_grad_norm_(
                self.agent.valueNet.parameters(),
                max_norm=0.5
            )
            
            self.agent.policyHead.optimizer.step()
            self.agent.valueNet.optimizer.step()
            # SRNet update if applicable
            if self.agent.SRNet is not None:
                sr_loss = self.agent.SRNet.compute_loss(batch)
                self.agent.SRNet.optimizer.zero_grad()
                sr_loss.backward()
                self.agent.SRNet.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        return {
            'policy_loss': total_policy_loss / self.config['ppo_epochs'],
            'value_loss': total_value_loss / self.config['ppo_epochs'],
            'returns_mean': returns.mean().item(),
        }


    
