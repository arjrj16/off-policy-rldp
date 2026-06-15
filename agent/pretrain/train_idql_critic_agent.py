"""
Offline IDQL critic (Q/V) pretraining.

Trains only critic_q / critic_v (plus the target_q EMA copy) of an
IDQLDiffusion model with IQL losses on offline chunk transitions
(StitchedChunkQLearningDataset), so the pretrained critics match the chunked
MDP the fine-tuner uses: Q(s_t, a_{t:t+H}), chunk-summed reward, gamma per
chunk. The diffusion BC actor is instantiated but never updated or used here
(IQL critic training needs only dataset actions), so this stage can run
independently of / in parallel with BC actor pretraining.

Checkpoints store the critic state dicts only and are consumed at fine-tuning
time via IDQLDiffusion(critic_path=...).

Standalone rather than a PreTrainAgent subclass: that base class hardwires a
single optimizer, an EMA over the full model, and BC-loss-shaped batches,
none of which apply to IQL critic training.
"""

import os
import random
import numpy as np
from omegaconf import OmegaConf
import torch
import hydra
import logging
import wandb

log = logging.getLogger(__name__)
from util.scheduler import CosineAnnealingWarmupRestarts
from agent.pretrain.train_agent import batch_to_device


class TrainIDQLCriticPretrainAgent:

    def __init__(self, cfg):
        super().__init__()
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

        # Full IDQL model; only its critics are trained here
        self.model = hydra.utils.instantiate(cfg.model)

        # Training params
        self.n_epochs = cfg.train.n_epochs
        self.batch_size = cfg.train.batch_size
        self.gamma = cfg.train.gamma
        self.critic_tau = cfg.train.critic_tau

        # Logging, checkpoints
        self.logdir = cfg.logdir
        self.checkpoint_dir = os.path.join(self.logdir, "checkpoint")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.log_freq = cfg.train.get("log_freq", 1)
        self.save_model_freq = cfg.train.save_model_freq

        # Dataset
        self.dataset_train = hydra.utils.instantiate(cfg.train_dataset)
        self.dataloader_train = torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            num_workers=4 if self.dataset_train.device == "cpu" else 0,
            shuffle=True,
            pin_memory=True if self.dataset_train.device == "cpu" else False,
        )

        # Optimizers / schedulers — mirror the fine-tuner's critic setup
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

    def run(self):
        cnt_update = 0
        for self.epoch in range(1, self.n_epochs + 1):
            loss_v_epoch = []
            loss_q_epoch = []
            for batch in self.dataloader_train:
                if self.dataset_train.device == "cpu":
                    batch = batch_to_device(batch, device=self.device)
                actions, conditions, rewards, dones = batch
                obs = {"state": conditions["state"]}
                next_obs = {"state": conditions["next_state"]}

                # Same update order and losses as the fine-tuner: V first
                # (expectile toward target-Q), then Q (TD toward r + gamma*V'),
                # then the target-Q polyak update.
                loss_v = self.model.loss_critic_v(obs, actions)
                self.critic_v_optimizer.zero_grad()
                loss_v.backward()
                self.critic_v_optimizer.step()

                # Offline chunks are never truncated mid-window (the dataset
                # drops windows whose next state would leave the trajectory),
                # so truncated is all-zero and dones carry within-chunk success.
                loss_q = self.model.loss_critic_q(
                    obs,
                    next_obs,
                    actions,
                    rewards.view(-1),
                    dones.view(-1),
                    torch.zeros_like(dones.view(-1)),
                    self.gamma,
                )
                self.critic_q_optimizer.zero_grad()
                loss_q.backward()
                self.critic_q_optimizer.step()

                self.model.update_target_critic(self.critic_tau)
                loss_v_epoch.append(loss_v.item())
                loss_q_epoch.append(loss_q.item())
                cnt_update += 1

            self.critic_v_lr_scheduler.step()
            self.critic_q_lr_scheduler.step()
            loss_v_mean = np.mean(loss_v_epoch)
            loss_q_mean = np.mean(loss_q_epoch)

            if self.epoch % self.save_model_freq == 0 or self.epoch == self.n_epochs:
                self.save_model()
            if self.epoch % self.log_freq == 0:
                log.info(
                    f"epoch {self.epoch} | loss v {loss_v_mean:8.4f} | "
                    f"loss q {loss_q_mean:8.4f} | updates {cnt_update}"
                )
                if self.use_wandb:
                    wandb.log(
                        {
                            "loss - critic v": loss_v_mean,
                            "loss - critic q": loss_q_mean,
                            "num updates": cnt_update,
                        },
                        step=self.epoch,
                    )

    def save_model(self):
        """Critic-only checkpoint, consumed by IDQLDiffusion(critic_path=...)."""
        data = {
            "epoch": self.epoch,
            "critic_q": self.model.critic_q.state_dict(),
            "critic_v": self.model.critic_v.state_dict(),
            "target_q": self.model.target_q.state_dict(),
        }
        savepath = os.path.join(self.checkpoint_dir, f"state_{self.epoch}.pt")
        torch.save(data, savepath)
        log.info(f"Saved critic checkpoint to {savepath}")
