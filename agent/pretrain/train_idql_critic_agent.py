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
from copy import deepcopy

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
        self.dataloader_val = None
        val_split = cfg.train.get("val_split", 0.0)
        if val_split and val_split > 0:
            # Episode-level holdout split (no leakage across train/val).
            # Group tuple indices by their episode start (start - num_before_start).
            episode_to_indices = {}
            for start, num_before_start in self.dataset_train.indices:
                ep_start = start - num_before_start
                episode_to_indices.setdefault(ep_start, []).append((start, num_before_start))

            episode_starts = list(episode_to_indices.keys())
            n_val_eps = int(len(episode_starts) * val_split)
            n_val_eps = max(1, n_val_eps) if len(episode_starts) > 1 else 0
            rng = random.Random(self.seed)
            val_eps = set(rng.sample(episode_starts, n_val_eps)) if n_val_eps > 0 else set()

            train_indices = []
            val_indices = []
            for ep_start, idx_list in episode_to_indices.items():
                if ep_start in val_eps:
                    val_indices.extend(idx_list)
                else:
                    train_indices.extend(idx_list)

            self.dataset_train.set_indices(train_indices)
            self.dataset_val = deepcopy(self.dataset_train)
            self.dataset_val.set_indices(val_indices)
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                num_workers=4 if self.dataset_val.device == "cpu" else 0,
                shuffle=False,
                pin_memory=True if self.dataset_val.device == "cpu" else False,
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

            for batch in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch = batch_to_device(batch, self.device)

                # Unpack batch (supports reward_to_go field)
                if len(batch) == 5:
                    actions, conditions, rewards, dones, _ = batch
                else:
                    actions, conditions, rewards, dones = batch

                obs = {"state": conditions["state"]}
                next_obs = {"state": conditions["next_state"]}

                # Update critic V
                loss_v = self.model.loss_critic_v(obs, actions)
                self.critic_v_optimizer.zero_grad()
                loss_v.backward()
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
                self.critic_q_optimizer.step()

                # Update target critic
                self.model.update_target_critic(self.critic_tau)

                loss_q_epoch.append(loss_q.item())
                loss_v_epoch.append(loss_v.item())

            # Update lr
            self.critic_q_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()

            # Validation
            loss_q_val = None
            loss_v_val = None
            if self.dataloader_val is not None:
                self.model.eval()
                loss_q_val_epoch = []
                loss_v_val_epoch = []
                with torch.no_grad():
                    for batch in self.dataloader_val:
                        if self.dataset_val.device == "cpu":
                            batch = batch_to_device(batch, self.device)
                        if len(batch) == 5:
                            actions, conditions, rewards, dones, _ = batch
                        else:
                            actions, conditions, rewards, dones = batch
                        obs = {"state": conditions["state"]}
                        next_obs = {"state": conditions["next_state"]}
                        loss_v = self.model.loss_critic_v(obs, actions)
                        loss_q = self.model.loss_critic_q(
                            obs,
                            next_obs,
                            actions,
                            rewards,
                            dones,
                            self.gamma,
                        )
                        loss_q_val_epoch.append(loss_q.item())
                        loss_v_val_epoch.append(loss_v.item())
                loss_q_val = (
                    float(np.mean(loss_q_val_epoch)) if loss_q_val_epoch else 0.0
                )
                loss_v_val = (
                    float(np.mean(loss_v_val_epoch)) if loss_v_val_epoch else 0.0
                )
                self.model.train()

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
                    log_payload = {
                        "loss - critic_q": avg_loss_q,
                        "loss - critic_v": avg_loss_v,
                    }
                    if loss_q_val is not None and loss_v_val is not None:
                        log_payload.update(
                            {
                                "loss - critic_q_val": loss_q_val,
                                "loss - critic_v_val": loss_v_val,
                            }
                        )
                    wandb.log(log_payload, step=self.epoch, commit=True)

            self.epoch += 1

    def save_model(self):
        data = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved model to {savepath}")
