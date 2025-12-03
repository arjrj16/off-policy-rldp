"""
Implicit diffusion Q-learning (IDQL) for diffusion policy.

"""

import logging
import torch
import einops
import copy

import torch.nn.functional as F

log = logging.getLogger(__name__)

from model.diffusion.diffusion_rwr import RWRDiffusion
from model.diffusion.diffusion import DiffusionModel


def expectile_loss(diff, expectile=0.8):
    weight = torch.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)


class IDQLDiffusion(RWRDiffusion):

    def __init__(
        self,
        actor,
        critic_q,
        critic_v,
        **kwargs,
    ):
        super().__init__(network=actor, **kwargs)
        self.critic_q = critic_q.to(self.device)
        self.target_q = copy.deepcopy(critic_q)
        self.critic_v = critic_v.to(self.device)

        # assign actor
        self.actor = self.network
        
        # Create frozen copy of BC actor for diagnostic support distance computation
        self.bc_actor_frozen = copy.deepcopy(actor).to(self.device)
        for param in self.bc_actor_frozen.parameters():
            param.requires_grad = False
        log.info("copied bc actor for support dist calc")

    # ---------- RL training ----------#

    def compute_advantages(self, obs, actions):

        # get current Q-function, stop gradient
        with torch.no_grad():
            current_q1, current_q2 = self.target_q(obs, actions)
        q = torch.min(current_q1, current_q2)

        # get the current V-function
        v = self.critic_v(obs).reshape(-1)

        # compute advantage
        adv = q - v
        return adv

    def loss_critic_v(self, obs, actions):
        adv = self.compute_advantages(obs, actions)

        # get the value loss
        v_loss = expectile_loss(adv).mean()
        return v_loss

    def loss_critic_q(self, obs, next_obs, actions, rewards, terminated, gamma):

        # get current Q-function
        current_q1, current_q2 = self.critic_q(obs, actions)

        # get the next V-function, stop gradient
        with torch.no_grad():
            next_v = self.critic_v(next_obs)

        # terminal state mask
        mask = 1 - terminated

        # flatten
        rewards = rewards.view(-1)
        next_v = next_v.view(-1)
        mask = mask.view(-1)

        # target value
        discounted_q = rewards + gamma * next_v * mask

        # Update critic
        q_loss = torch.mean((current_q1 - discounted_q) ** 2) + torch.mean(
            (current_q2 - discounted_q) ** 2
        )
        return q_loss

    def update_target_critic(self, tau):
        for target_param, source_param in zip(
            self.target_q.parameters(), self.critic_q.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - tau) + source_param.data * tau
            )

    # override
    def p_losses(
        self,
        x_start,
        cond,
        t,
    ):
        """not reward-weighted, same as diffusion.py"""
        device = x_start.device

        # Forward process
        noise = torch.randn_like(x_start, device=device)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Predict
        x_recon = self.network(x_noisy, t, cond=cond)

        # Loss with mask
        if self.predict_epsilon:
            loss = F.mse_loss(x_recon, noise)
        else:
            loss = F.mse_loss(x_recon, x_start)
        return loss.mean()

    # ---------- Sampling ----------#

    def _compute_pairwise_support_distance(self, samples_expanded, act_steps=None):
        """Compute support distance using pairwise distance among candidates.
        
        For each candidate, compute min L2 distance to other candidates.
        This is a cheap proxy for "how unusual" a sample is.
        
        Args:
            samples_expanded: (S, B, H, A) candidate samples
            act_steps: if provided, only use first act_steps for distance computation
            
        Returns:
            support_dist: (S, B) distance for each candidate
        """
        S, B, H, A = samples_expanded.shape
        
        # Optionally truncate to act_steps
        if act_steps is not None:
            samples_for_dist = samples_expanded[:, :, :act_steps, :]  # (S, B, act_steps, A)
        else:
            samples_for_dist = samples_expanded
        
        # Flatten action dimensions: (S, B, act_steps * A)
        samples_flat = samples_for_dist.reshape(S, B, -1)
        
        # Compute pairwise distances: for each batch, compute S x S distance matrix
        # samples_flat: (S, B, D) where D = act_steps * A
        # We want: dist[i, j, b] = ||samples_flat[i, b] - samples_flat[j, b]||
        
        # Expand for broadcasting: (S, 1, B, D) - (1, S, B, D) -> (S, S, B, D)
        diff = samples_flat[:, None, :, :] - samples_flat[None, :, :, :]  # (S, S, B, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (S, S, B)
        dist = torch.sqrt(dist_sq + 1e-8)  # (S, S, B)
        
        # For each sample i, find min distance to any other sample j != i
        # Set diagonal to inf so we don't pick self
        inf_diag = torch.eye(S, device=dist.device).unsqueeze(-1) * 1e9  # (S, S, 1)
        dist_masked = dist + inf_diag  # (S, S, B)
        
        # Min over j dimension gives us the nearest neighbor distance for each sample
        support_dist, _ = dist_masked.min(dim=1)  # (S, B)
        
        return support_dist

    def _compute_bc_support_distance(self, samples_expanded, cond, bc_num_samples=32, act_steps=None):
        """Compute support distance by sampling from frozen BC actor.
        
        More accurate than pairwise but slower.
        
        Args:
            samples_expanded: (S, B, H, A) candidate samples
            cond: conditioning dict with state
            bc_num_samples: number of BC samples to draw for comparison
            act_steps: if provided, only use first act_steps for distance computation
            
        Returns:
            support_dist: (S, B) min distance to BC samples for each candidate
        """
        S, B, H, A = samples_expanded.shape
        
        # Sample from frozen BC
        bc_samples = self.sample_from_frozen_bc(cond, num_samples=bc_num_samples, deterministic=False)
        # bc_samples: (K, B, H, A) where K = bc_num_samples
        K = bc_samples.shape[0]
        
        # Optionally truncate to act_steps
        if act_steps is not None:
            samples_for_dist = samples_expanded[:, :, :act_steps, :]  # (S, B, act_steps, A)
            bc_for_dist = bc_samples[:, :, :act_steps, :]  # (K, B, act_steps, A)
        else:
            samples_for_dist = samples_expanded
            bc_for_dist = bc_samples
        
        # Flatten: (S, B, D) and (K, B, D)
        samples_flat = samples_for_dist.reshape(S, B, -1)
        bc_flat = bc_for_dist.reshape(K, B, -1)
        
        # Compute distances: (S, K, B)
        diff = samples_flat[:, None, :, :] - bc_flat[None, :, :, :]  # (S, K, B, D)
        dist_sq = (diff ** 2).sum(dim=-1)  # (S, K, B)
        dist = torch.sqrt(dist_sq + 1e-8)  # (S, K, B)
        
        # Min over BC samples
        support_dist, _ = dist.min(dim=1)  # (S, B)
        
        return support_dist

    # override
    @torch.no_grad()
    def forward(
        self,
        cond,
        deterministic=False,
        num_sample=10,
        critic_hyperparam=0.7,  # sampling weight for implicit policy
        use_expectile_exploration=True,
        return_diagnostics=False,
        # Support filtering params
        support_filter_enabled=False,
        support_topk=None,  # If set, keep only top-K candidates by smallest support distance
        support_filter_mode="pairwise",  # "pairwise" or "bc"
        support_bc_num_samples=32,  # Number of BC samples for "bc" mode
        support_act_steps=None,  # Act steps for distance computation (defaults to all)
    ):
        """assume state-only, no rgb in cond
        
        Args:
            return_diagnostics: (samples, diagnostics_dict) where
                diagnostics_dict contains:
                - all_samples: (S, B, H, A) all candidate samples before Q-selection
                - all_q: (S, B) Q-values for all samples
                - chosen_indices: (B,) indices of chosen samples
                - support_dist: (S, B) support distances (if filter enabled)
                - filtered_mask: (S, B) bool mask of which candidates survived filtering
            support_filter_enabled: Whether to filter candidates by support distance
            support_topk: Keep only top-K candidates by smallest support distance.
                         If None, uses num_sample (no filtering even if enabled).
            support_filter_mode: "pairwise" (cheap, uses inter-candidate distance) or
                                "bc" (accurate, samples from frozen BC)
            support_bc_num_samples: Number of BC samples when mode="bc"
            support_act_steps: Only use first N action steps for distance computation
        """
        # repeat obs num_sample times along dim 0
        cond_shape_repeat_dims = tuple(1 for _ in cond["state"].shape)
        B, T, D = cond["state"].shape
        S = num_sample
        cond_repeat = cond["state"][None].repeat(num_sample, *cond_shape_repeat_dims)
        cond_repeat = cond_repeat.view(-1, T, D)  # [B*S, T, D]

        # for eval, use less noisy samples --- there is still DDPM noise, but final action uses small min_sampling_std
        samples = super(IDQLDiffusion, self).forward(
            {"state": cond_repeat},
            deterministic=deterministic,
        )
        _, H, A = samples.shape
        samples_expanded = samples.view(S, B, H, A)

        # get current Q-function
        current_q1, current_q2 = self.target_q({"state": cond_repeat}, samples)
        q = torch.min(current_q1, current_q2)
        q = q.view(S, B)
        
        # --- Support filtering ---
        support_dist = None
        filtered_mask = None
        q_filtered = q  # Will be modified if filtering is applied
        
        if support_filter_enabled and support_topk is not None and support_topk < S:
            # Compute support distances
            if support_filter_mode == "bc":
                support_dist = self._compute_bc_support_distance(
                    samples_expanded, cond, 
                    bc_num_samples=support_bc_num_samples,
                    act_steps=support_act_steps
                )
            else:  # pairwise
                support_dist = self._compute_pairwise_support_distance(
                    samples_expanded, act_steps=support_act_steps
                )
            
            # Create filter mask: keep top-K by smallest support distance
            # For each batch, find the K samples with smallest support_dist
            _, topk_indices = support_dist.topk(k=support_topk, dim=0, largest=False)  # (K, B)
            
            # Create boolean mask (S, B)
            filtered_mask = torch.zeros_like(q, dtype=torch.bool)
            batch_indices = torch.arange(B, device=q.device).unsqueeze(0).expand(support_topk, -1)
            filtered_mask[topk_indices, batch_indices] = True
            
            # Set Q of filtered-out candidates to -inf so they won't be selected
            q_filtered = q.clone()
            q_filtered[~filtered_mask] = float('-inf')

        # Use argmax (with support filtering applied via q_filtered)
        if deterministic or (not use_expectile_exploration):
            # gather the best sample -- filter out suboptimal Q during inference
            best_indices = q_filtered.argmax(0)

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = best_indices[None, :, None, None]  # [1, B, 1, 1]
            sample_indices = sample_indices.repeat(S, 1, H, A)

            samples_best = torch.gather(samples_expanded, 0, sample_indices)
            chosen_indices = best_indices
        # Sample as an implicit policy for exploration
        else:
            # get the current value function for probabilistic exploration
            current_v = self.critic_v({"state": cond_repeat})
            v = current_v.view(S, B)
            adv = q - v

            # expectile exploration policy
            tau_weights = torch.where(adv > 0, critic_hyperparam, 1 - critic_hyperparam)
            
            # Apply support filter to weights: zero out filtered candidates
            if filtered_mask is not None:
                tau_weights = tau_weights * filtered_mask.float()
                # Handle case where all candidates are filtered for a batch element
                # (shouldn't happen with topk, but be safe)
                tau_weights = tau_weights + 1e-8
            
            tau_weights = tau_weights / tau_weights.sum(0)  # normalize

            # select a sample from DP probabilistically -- sample index per batch and compile
            sample_indices_multinomial = torch.multinomial(tau_weights.T, 1)  # [B, 1]
            chosen_indices = sample_indices_multinomial.squeeze(1)  # [B]

            # dummy dimension @ dim 0 for batched indexing
            sample_indices = sample_indices_multinomial[None, :, None]  # [1, B, 1, 1]
            sample_indices = sample_indices.repeat(S, 1, H, A)

            samples_best = torch.gather(samples_expanded, 0, sample_indices)

        # squeeze dummy dimension
        samples = samples_best[0]
        
        if return_diagnostics:
            diagnostics = {
                "all_samples": samples_expanded,  # (S, B, H, A)
                "all_q": q,  # (S, B) - original Q values
                "chosen_indices": chosen_indices,  # (B,)
            }
            if support_dist is not None:
                diagnostics["support_dist"] = support_dist  # (S, B)
            if filtered_mask is not None:
                diagnostics["filtered_mask"] = filtered_mask  # (S, B)
                diagnostics["num_filtered"] = filtered_mask.sum(dim=0).float().mean().item()
            return samples, diagnostics
        return samples
    
    @torch.no_grad()
    def sample_from_frozen_bc(self, cond, num_samples=64, deterministic=False):
        """Sample from frozen BC actor without Q-filtering."""
        B, T, D = cond["state"].shape
        S = num_samples
        
        # Repeat cond for num_samples
        cond_repeat = cond["state"][None].repeat(num_samples, *(1,) * len(cond["state"].shape))
        cond_repeat = cond_repeat.view(-1, T, D)  # [B*S, T, D]
        
        # Temporarily swap network to frozen BC actor
        original_network = self.network
        self.network = self.bc_actor_frozen
        
        try:
            from model.diffusion.sampling import make_timesteps
            device = self.betas.device
            x = torch.randn((B * S, self.horizon_steps, self.action_dim), device=device)
            if self.use_ddim:
                t_all = self.ddim_t
            else:
                t_all = list(reversed(range(self.denoising_steps)))
            for i, t in enumerate(t_all):
                t_b = make_timesteps(B * S, t, device)
                index_b = make_timesteps(B * S, i, device)
                mean, logvar = self.p_mean_var(
                    x=x,
                    t=t_b,
                    cond={"state": cond_repeat},
                    index=index_b,
                )
                std = torch.exp(0.5 * logvar)
                if self.use_ddim:
                    std = torch.zeros_like(std) if deterministic else std
                else:
                    if deterministic and t == 0:
                        std = torch.zeros_like(std)
                    else:
                        std = torch.clip(std, min=1e-3)
                noise = torch.randn_like(x).clamp_(
                    -self.randn_clip_value, self.randn_clip_value
                )
                x = mean + std * noise
                if self.final_action_clip_value is not None and i == len(t_all) - 1:
                    x = torch.clamp(
                        x, -self.final_action_clip_value, self.final_action_clip_value
                    )
            _, H, A = x.shape
            bc_samples = x.view(S, B, H, A)
        finally:
            self.network = original_network
        
        return bc_samples