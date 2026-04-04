"""
Offline pretraining for IDQL critics (Q and V) using demonstration data.
"""

import os
import random
import logging

import numpy as np
import torch
import hydra
import wandb
from omegaconf import OmegaConf

from util.scheduler import CosineAnnealingWarmupRestarts
from agent.pretrain.train_agent import batch_to_device

log = logging.getLogger(__name__)


class TrainIDQLCriticPretrainAgent:
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = cfg.device
        self.seed = cfg.get("seed", 42)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Wandb
        self.use_wandb = cfg.wandb is not None
        if cfg.wandb is not None:
            wandb.init(
                entity=cfg.wandb.entity,
                project=cfg.wandb.project,
                name=cfg.wandb.run,
                config=OmegaConf.to_container(cfg, resolve=True),
            )

        # Build model (includes actor + critics)
        self.model = hydra.utils.instantiate(cfg.model)
        self.model.actor.eval()
        for param in self.model.actor.parameters():
            param.requires_grad_(False)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.critic_tau = cfg.train.critic_tau
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq
        self.max_grad_norm = cfg.train.get("max_grad_norm", 1.0)

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Build dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )

        # Optimizers
        self.critic_q_optimizer = torch.optim.AdamW(
            self.model.critic_q.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_v_optimizer = torch.optim.AdamW(
            self.model.critic_v.parameters(),
            lr=cfg.train.critic_lr,
            weight_decay=cfg.train.critic_weight_decay,
        )
        self.critic_q_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_q_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
        self.critic_v_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_v_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
        )

        self.epoch = 1

    def run(self):
        for _ in range(self.n_epochs):
            loss_q_epoch = []
            loss_v_epoch = []
            reward_present_epoch = False
            reward_batch_count = 0
            total_batch_count = 0
            reward_step_count = 0
            total_step_count = 0
            loss_q_reward_batches = []
            loss_q_nonreward_batches = []

            for batch in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch = batch_to_device(batch, self.device)

                # Unpack batch (supports reward_to_go field)
                if len(batch) == 5:
                    actions, conditions, rewards, dones, _ = batch
                else:
                    actions, conditions, rewards, dones = batch

                reward_present_epoch = reward_present_epoch or (rewards > 0).any()
                batch_has_reward = (rewards > 0).any().item()
                reward_batch_count += int(batch_has_reward)
                total_batch_count += 1
                reward_step_count += int((rewards > 0).sum().item())
                total_step_count += int(rewards.numel())

                obs = {"state": conditions["state"]}
                next_obs = {"state": conditions["next_state"]}

                # Update critic V
                loss_v = self.model.loss_critic_v(obs, actions)
                self.critic_v_optimizer.zero_grad()
                loss_v.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic_v.parameters(), self.max_grad_norm
                )
                self.critic_v_optimizer.step()

                # Update critic Q
                loss_q = self.model.loss_critic_q(
                    obs,
                    next_obs,
                    actions,
                    rewards,
                    dones,
                    self.gamma,
                )
                self.critic_q_optimizer.zero_grad()
                loss_q.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.critic_q.parameters(), self.max_grad_norm
                )
                self.critic_q_optimizer.step()

                # Update target critic
                self.model.update_target_critic(self.critic_tau)

                loss_q_epoch.append(loss_q.item())
                loss_v_epoch.append(loss_v.item())
                if batch_has_reward:
                    loss_q_reward_batches.append(loss_q.item())
                else:
                    loss_q_nonreward_batches.append(loss_q.item())

            # Update lr
            self.critic_q_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()

            # Save model
            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()

            # Log
            if self.epoch % self.log_freq == 0:
                avg_loss_q = float(np.mean(loss_q_epoch)) if loss_q_epoch else 0.0
                avg_loss_v = float(np.mean(loss_v_epoch)) if loss_v_epoch else 0.0
                log.info(
                    f"{self.epoch}: critic_q {avg_loss_q:8.4f} | critic_v {avg_loss_v:8.4f}"
                )
                if self.use_wandb:
                    reward_batch_fraction = (
                        reward_batch_count / total_batch_count
                        if total_batch_count > 0
                        else 0.0
                    )
                    reward_step_fraction = (
                        reward_step_count / total_step_count if total_step_count > 0 else 0.0
                    )
                    loss_q_reward = (
                        float(np.mean(loss_q_reward_batches))
                        if loss_q_reward_batches
                        else 0.0
                    )
                    loss_q_nonreward = (
                        float(np.mean(loss_q_nonreward_batches))
                        if loss_q_nonreward_batches
                        else 0.0
                    )
                    wandb.log(
                        {
                            "loss - critic_q": avg_loss_q,
                            "loss - critic_v": avg_loss_v,
                            "reward_present": float(reward_present_epoch),
                            "reward_batch_fraction": reward_batch_fraction,
                            "reward_step_fraction": reward_step_fraction,
                            "loss_q_reward_batches": loss_q_reward,
                            "loss_q_nonreward_batches": loss_q_nonreward,
                        },
                        step=self.epoch,
                        commit=True,
                    )

            self.epoch += 1

    def save_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
