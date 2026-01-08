"""
Implicit Q-learning (IDQL) for flow-matching policies.

This mirrors `model/diffusion/diffusion_idql.py` but replaces the diffusion
sampler / loss with conditional flow matching (velocity field + ODE sampler).
"""
import copy
import logging

import torch

log = logging.getLogger(__name__)

from model.flow_matching.flow_matching import FlowMatchingModel


def expectile_loss(diff, expectile: float = 0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IDQLFlowMatching(FlowMatchingModel):
    def __init__(
        self,
        actor,
        critic_q,
        critic_v,
        mask_truncated: bool = False,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.critic_q = critic_q.to(self.device)
        self.target_q = copy.deepcopy(critic_q)
        self.critic_v = critic_v.to(self.device)

        # alias (for parity with diffusion code)
        self.actor = self.network

        # true = mask bootstrap r of trajs that are truncated and terminated, false = mask only terminated trajs
        self.mask_truncated = mask_truncated

        # frozen BC copy for diagnostics (support distance)
        self.bc_actor_frozen = copy.deepcopy(actor).to(self.device)
        for p in self.bc_actor_frozen.parameters():
            p.requires_grad = False
        log.info("copied bc actor for support dist calc")

    # ---------- RL training ----------#

    def compute_advantages(self, obs, actions):
        with torch.no_grad():
            current_q1, current_q2 = self.target_q(obs, actions)
        q = torch.min(current_q1, current_q2)
        v = self.critic_v(obs).reshape(-1)
        return q - v

    def loss_critic_v(self, obs, actions):
        adv = self.compute_advantages(obs, actions)
        return expectile_loss(adv).mean()

    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, truncated, gamma):
        current_q1, current_q2 = self.critic_q(obs, actions)
        with torch.no_grad():
            next_v = self.critic_v(next_obs)

        if self.mask_truncated:
            mask = 1 - (terminated.bool() | truncated.bool()).float()
        else:
            mask = 1 - terminated

        rewards = rewards.view(-1)
        next_v = next_v.view(-1)
        mask = mask.view(-1)

        discounted_q = rewards + gamma * next_v * mask
        q_loss = torch.mean((current_q1 - discounted_q) ** 2) + torch.mean((current_q2 - discounted_q) ** 2)
        return q_loss

    def update_target_critic(self, tau: float):
        for target_param, source_param in zip(self.target_q.parameters(), self.critic_q.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + source_param.data * tau)

    # ---------- Sampling (Q-filtering) ----------#

    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic: bool = False,
        num_sample: int = 10,
        critic_hyperparam: float = 0.7,
        use_expectile_exploration: bool = True,
        return_diagnostics: bool = False,
        use_bc_warmup: bool = False,
    ):
        """
        Args:
            return_diagnostics: (samples, diagnostics_dict) where diagnostics contain:
              - all_samples: (S, B, H, A)
              - all_q: (S, B)
              - chosen_indices: (B,)
        """
        if use_bc_warmup:
            B, T, D = cond["state"].shape
            bc_samples = self.sample_from_frozen_bc(cond, num_samples=1, deterministic=deterministic)  # (1,B,H,A)
            samples = bc_samples[0]

            if return_diagnostics:
                bc_samples_multi = self.sample_from_frozen_bc(
                    cond, num_samples=num_sample, deterministic=deterministic
                )  # (S,B,H,A)
                H, A = bc_samples_multi.shape[2:]
                cond_repeat = cond["state"][None].repeat(num_sample, *(1,) * len(cond["state"].shape))
                cond_repeat = cond_repeat.view(-1, T, D)  # (S*B, T, D)
                bc_samples_flat = bc_samples_multi.view(-1, H, A)  # (S*B, H, A)
                current_q1, current_q2 = self.target_q({"state": cond_repeat}, bc_samples_flat)
                q = torch.min(current_q1, current_q2).view(num_sample, B)
                chosen_indices = torch.zeros(B, dtype=torch.long, device=self.device)
                diagnostics = {"all_samples": bc_samples_multi, "all_q": q, "chosen_indices": chosen_indices}
                return samples, diagnostics
            return samples

        # repeat obs S times
        B, T, D = cond["state"].shape
        S = int(num_sample)
        cond_repeat = cond["state"][None].repeat(S, *(1,) * len(cond["state"].shape)).view(-1, T, D)  # (S*B,T,D)

        # generate candidates (S*B,H,A)
        samples = super().forward({"state": cond_repeat}, deterministic=deterministic)
        _, H, A = samples.shape

        # score candidates
        current_q1, current_q2 = self.target_q({"state": cond_repeat}, samples)
        q = torch.min(current_q1, current_q2).view(S, B)

        if deterministic or (not use_expectile_exploration):
            best_indices = q.argmax(0)  # (B,)
            samples_expanded = samples.view(S, B, H, A)
            idx = best_indices[None, :, None, None].repeat(S, 1, H, A)
            samples_best = torch.gather(samples_expanded, 0, idx)
            chosen_indices = best_indices
        else:
            current_v = self.critic_v({"state": cond_repeat}).view(S, B)
            adv = q - current_v

            tau_weights = torch.where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
            tau_weights = tau_weights / tau_weights.sum(0)

            sample_indices_multinomial = torch.multinomial(tau_weights.T, 1)  # (B,1)
            chosen_indices = sample_indices_multinomial.squeeze(1)

            samples_expanded = samples.view(S, B, H, A)
            idx = sample_indices_multinomial[None, :, None].repeat(S, 1, H, A)
            samples_best = torch.gather(samples_expanded, 0, idx)

        samples = samples_best[0]  # (B,H,A)

        if return_diagnostics:
            diagnostics = {"all_samples": samples_expanded, "all_q": q, "chosen_indices": chosen_indices}
            return samples, diagnostics
        return samples

    @torch.no_grad()
    def sample_from_frozen_bc(self, cond, num_samples: int = 64, deterministic: bool = False):
        B, T, D = cond["state"].shape
        S = int(num_samples)

        cond_repeat = cond["state"][None].repeat(S, *(1,) * len(cond["state"].shape)).view(-1, T, D)  # (S*B,T,D)
        x = self.sample(cond={"state": cond_repeat}, deterministic=deterministic, network_override=self.bc_actor_frozen)
        _, H, A = x.shape
        return x.view(S, B, H, A)

