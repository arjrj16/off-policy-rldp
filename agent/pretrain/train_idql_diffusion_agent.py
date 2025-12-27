"""
Offline IDQL pre-training for diffusion policy.

This script trains:
1. Actor (diffusion policy): Behavior cloning loss
2. Critic Q: Bellman backup with rewards
3. Critic V: Expectile regression against Q

This allows the Q and V functions to be pre-trained offline before finetuning,
giving better initialization than random Q/V networks.

Requires a dataset with rewards (see relabel_robomimic_rewards.py).
"""

import logging
import wandb
import numpy as np
import torch

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.pretrain.train_agent import PreTrainAgent, batch_to_device
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainIDQLDiffusionAgent(PreTrainAgent):
    """
    Offline IDQL pretraining agent.
    
    Trains actor (BC) + critic Q + critic V from an offline dataset with rewards.
    """

    def __init__(self, cfg):
        # Initialize base class (this builds the model and dataset)
        super().__init__(cfg)
        
        # IDQL specific parameters
        self.gamma = cfg.train.gamma
        self.expectile = cfg.train.get("expectile", 0.7)
        self.critic_tau = cfg.train.get("critic_tau", 0.005)
        
        # Separate optimizers for actor and critics
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.get("actor_weight_decay", 1e-6),
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.lr_scheduler.min_lr,
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        
        self.critic_q_optimizer = torch.optim.AdamW(
            self.model.critic_q.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.get("critic_weight_decay", 0),
        )
        self.critic_q_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_q_optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.lr_scheduler.get("critic_min_lr", cfg.train.lr_scheduler.min_lr),
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        
        self.critic_v_optimizer = torch.optim.AdamW(
            self.model.critic_v.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.get("critic_weight_decay", 0),
        )
        self.critic_v_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_v_optimizer,
            first_cycle_steps=cfg.train.lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.lr_scheduler.get("critic_min_lr", cfg.train.lr_scheduler.min_lr),
            warmup_steps=cfg.train.lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        
        # Number of critic updates per actor update
        self.critic_updates_per_actor = cfg.train.get("critic_updates_per_actor", 1)
        
        # Reward scaling
        self.scale_reward_factor = cfg.train.get("scale_reward_factor", 1.0)
        
        # Action horizon for Q function
        self.act_steps = cfg.get("act_steps", cfg.horizon_steps)
        
        log.info(f"Offline IDQL pretraining with gamma={self.gamma}, expectile={self.expectile}")
        log.info(f"Actor LR: {cfg.train.actor_lr}, Critic LR: {cfg.train.critic_lr}")

    def expectile_loss(self, diff, expectile=None):
        """Asymmetric squared loss for expectile regression."""
        if expectile is None:
            expectile = self.expectile
        weight = torch.where(diff > 0, expectile, (1 - expectile))
        return weight * (diff ** 2)

    def loss_critic_v(self, obs, actions):
        """
        Compute V loss using expectile regression.
        V(s) should be the expectile of Q(s, a).
        """
        with torch.no_grad():
            # Get Q values from target network
            q1, q2 = self.model.target_q(obs, actions)
            q = torch.min(q1, q2)
        
        # Get V prediction
        v = self.model.critic_v(obs).view(-1)
        
        # Expectile loss: V should be high when Q is high
        adv = q - v
        v_loss = self.expectile_loss(adv).mean()
        
        return v_loss, {"v_mean": v.mean().item(), "adv_mean": adv.mean().item()}

    def loss_critic_q(self, obs, next_obs, actions, rewards, dones):
        """
        Compute Q loss using Bellman backup.
        Q(s, a) = r + gamma * V(s') * (1 - done)
        """
        # Get current Q predictions
        q1, q2 = self.model.critic_q(obs, actions)
        
        with torch.no_grad():
            # Get next V from V network (not target)
            next_v = self.model.critic_v(next_obs).view(-1)
            
            # Bellman target
            target_q = rewards + self.gamma * next_v * (1 - dones)
        
        # MSE loss for both Q networks
        q_loss = torch.mean((q1 - target_q) ** 2) + torch.mean((q2 - target_q) ** 2)
        
        return q_loss, {
            "q1_mean": q1.mean().item(),
            "q2_mean": q2.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def run(self):
        timer = Timer()
        self.epoch = 1
        cnt_batch = 0
        
        for _ in range(self.n_epochs):
            # Track losses for logging
            loss_actor_epoch = []
            loss_critic_q_epoch = []
            loss_critic_v_epoch = []
            
            for batch_train in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch_train = batch_to_device(batch_train)
                
                self.model.train()
                
                # Extract batch data
                # batch_train is a Transition namedtuple: (actions, conditions, rewards, dones)
                actions = batch_train.actions  # (B, H, A)
                conditions = batch_train.conditions  # dict with "state" and "next_state"
                rewards = batch_train.rewards.view(-1) * self.scale_reward_factor  # (B,)
                dones = batch_train.dones.view(-1)  # (B,)
                
                obs = {"state": conditions["state"]}  # (B, T, D)
                next_obs = {"state": conditions["next_state"]}  # (B, T, D)
                
                # Only use first act_steps of actions for Q function
                actions_for_q = actions[:, :self.act_steps, :]
                
                # ----- Update Critic V -----
                loss_v, v_info = self.loss_critic_v(obs, actions_for_q)
                self.critic_v_optimizer.zero_grad()
                loss_v.backward()
                self.critic_v_optimizer.step()
                loss_critic_v_epoch.append(loss_v.item())
                
                # ----- Update Critic Q -----
                loss_q, q_info = self.loss_critic_q(
                    obs, next_obs, actions_for_q, rewards, dones
                )
                self.critic_q_optimizer.zero_grad()
                loss_q.backward()
                self.critic_q_optimizer.step()
                loss_critic_q_epoch.append(loss_q.item())
                
                # Update target Q network
                self.model.update_target_critic(self.critic_tau)
                
                # ----- Update Actor (BC) -----
                loss_actor = self.model.loss(actions, obs)
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()
                loss_actor_epoch.append(loss_actor.item())
                
                # Update EMA for actor
                if cnt_batch % self.update_ema_freq == 0:
                    self.step_ema()
                cnt_batch += 1
            
            # Compute epoch losses
            loss_actor = np.mean(loss_actor_epoch)
            loss_critic_q = np.mean(loss_critic_q_epoch)
            loss_critic_v = np.mean(loss_critic_v_epoch)
            
            # Update learning rates
            self.actor_lr_scheduler.step()
            self.critic_q_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()
            
            # Save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
            
            # Log
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"{self.epoch}: actor {loss_actor:8.4f} | Q {loss_critic_q:8.4f} | V {loss_critic_v:8.4f} | t:{timer():8.4f}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "loss - actor": loss_actor,
                            "loss - critic_q": loss_critic_q,
                            "loss - critic_v": loss_critic_v,
                        },
                        step=self.epoch,
                        commit=True,
                    )
            
            self.epoch += 1

    def save_model(self):
        """
        Saves model (actor + critics) and EMA to disk.
        """
        import os
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
