"""
Implicit diffusion Q-learning (IDQL) trainer for diffusion policy.

Do not support pixel input right now.

"""

import os
import pickle
import einops
import numpy as np
import torch
import logging
import wandb
from copy import deepcopy

log = logging.getLogger(__name__)
from util.timer import Timer
from collections import deque
from agent.finetune.train_agent import TrainAgent
from util.scheduler import CosineAnnealingWarmupRestarts


class TrainIDQLDiffusionAgent(TrainAgent):

    def __init__(self, cfg):
        super().__init__(cfg)

        # note the discount factor gamma here is applied to reward every act_steps, instead of every env step
        self.gamma = cfg.train.gamma

        # Wwarm up period for critic before actor updates
        self.n_critic_warmup_itr = cfg.train.n_critic_warmup_itr

        # Optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.model.actor.parameters(),
            lr=cfg.train.actor_lr,
            weight_decay=cfg.train.actor_weight_decay,
        )
        self.actor_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.actor_optimizer,
            first_cycle_steps=cfg.train.actor_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.actor_lr,
            min_lr=cfg.train.actor_lr_scheduler.min_lr,
            warmup_steps=cfg.train.actor_lr_scheduler.warmup_steps,
            gamma=1.0,
        )
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
        self.critic_v_lr_scheduler = CosineAnnealingWarmupRestarts(
            self.critic_v_optimizer,
            first_cycle_steps=cfg.train.critic_lr_scheduler.first_cycle_steps,
            cycle_mult=1.0,
            max_lr=cfg.train.critic_lr,
            min_lr=cfg.train.critic_lr_scheduler.min_lr,
            warmup_steps=cfg.train.critic_lr_scheduler.warmup_steps,
            gamma=1.0,
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

        # Buffer size
        self.buffer_size = cfg.train.buffer_size

        # Actor params
        self.use_expectile_exploration = cfg.train.use_expectile_exploration

        # Scaling reward
        self.scale_reward_factor = cfg.train.scale_reward_factor

        # Updates
        self.replay_ratio = cfg.train.replay_ratio
        self.critic_tau = cfg.train.critic_tau

        # Whether to use deterministic mode when sampling at eval
        self.eval_deterministic = cfg.train.get("eval_deterministic", False)

        # Sampling
        self.num_sample = cfg.train.eval_sample_num
        
        # Diagnostic tracking
        self.diagnostic_enabled = cfg.train.get("diagnostic", {}).get("enabled", False)
        self.diagnostic_env_idx = cfg.train.get("diagnostic", {}).get("env_idx", 0)
        self.diagnostic_bc_num_samples = cfg.train.get("diagnostic", {}).get("bc_num_samples", 64)
        if self.diagnostic_enabled:
            log.info(f"q tracking enabled for environment {self.diagnostic_env_idx}")

    def run(self):

        # make a FIFO replay buffer for obs, action, and reward
        obs_buffer = deque(maxlen=self.buffer_size)
        next_obs_buffer = deque(maxlen=self.buffer_size)
        action_buffer = deque(maxlen=self.buffer_size)
        reward_buffer = deque(maxlen=self.buffer_size)
        terminated_buffer = deque(maxlen=self.buffer_size)

        # Start training loop
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        
        diag_current_episode = None
        diag_episode_data = []
        
        while self.itr < self.n_train_itr:

            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            forced_reset_this_itr = self.reset_at_iteration or eval_mode or last_itr_eval
            if forced_reset_this_itr:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
                # If we forced a reset, any ongoing diagnostic episode is aborted (truncated)
                if self.diagnostic_enabled and diag_current_episode is not None:
                    # Tag and drop aborted episode (don't include in metrics - it's corrupted)
                    diag_current_episode = None
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            
            diag_episode_data_this_itr = []

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    cond = {
                        "state": torch.from_numpy(prev_obs_venv["state"])
                        .float()
                        .to(self.device)
                    }
                    
                    # Get action with diagnostics if enabled
                    if self.diagnostic_enabled:
                        result = self.model(
                            cond=cond,
                            deterministic=eval_mode and self.eval_deterministic,
                            num_sample=self.num_sample,
                            use_expectile_exploration=self.use_expectile_exploration,
                            return_diagnostics=True,
                        )
                        samples, diag = result
                        samples = samples.cpu().numpy()
                    else:
                        samples = (
                            self.model(
                                cond=cond,
                                deterministic=eval_mode and self.eval_deterministic,
                                num_sample=self.num_sample,
                                use_expectile_exploration=self.use_expectile_exploration,
                            )
                            .cpu()
                            .numpy()
                        )  # n_env x horizon x act
                        diag = None
                    
                action_venv = samples[:, : self.act_steps]
                
                # Compute diagnostics for tracked environment
                if self.diagnostic_enabled and diag is not None and self.diagnostic_env_idx < self.n_envs:
                    if firsts_trajs[step, self.diagnostic_env_idx] == 1:
                        if diag_current_episode is not None:
                            log.warning(f"Starting new episode but diag_current_episode exists at step {step}, itr {self.itr}")
                            diag_current_episode = None
                        diag_current_episode = {
                            "q_start": None,
                            "rewards": [],
                            "support_distances": [],
                            "q_values": [],
                            "step": step,
                            "itr": self.itr,
                        }
                    
                    # Check if episode is still active (not done from previous step)
                    env_was_done = done_venv[self.diagnostic_env_idx] if step > 0 else False
                    
                    # Compute Q-value for chosen action (for env 0) - only if episode is active
                    if diag_current_episode is not None and not env_was_done:
                        # Get Q-value of chosen action
                        chosen_idx = diag["chosen_indices"][self.diagnostic_env_idx].item()
                        q_chosen = diag["all_q"][chosen_idx, self.diagnostic_env_idx].item()
                        
                        # Store Q at episode start
                        if diag_current_episode["q_start"] is None:
                            diag_current_episode["q_start"] = q_chosen
                        
                        # Sample from frozen BC to compute support distance
                        cond_env0 = {
                            "state": cond["state"][self.diagnostic_env_idx:self.diagnostic_env_idx+1]  # (1, T, D)
                        }
                        bc_samples = self.model.sample_from_frozen_bc(
                            cond_env0,
                            num_samples=self.diagnostic_bc_num_samples,
                            deterministic=False,
                        )  # (num_samples, 1, H, A)
                        
                        # Get chosen action for env 0
                        chosen_action = torch.from_numpy(samples[self.diagnostic_env_idx]).float().to(self.device)  # (H, A)
                        chosen_action = chosen_action[:self.act_steps]  # Only use act_steps
                        
                        # Compute L2 distance to nearest BC sample
                        bc_samples_env0 = bc_samples[:, 0, :self.act_steps, :]  # (num_samples, act_steps, A)
                        chosen_action_expanded = chosen_action.unsqueeze(0)  # (1, act_steps, A)
                        
                        l2_distances = ((bc_samples_env0 - chosen_action_expanded) ** 2).sum(dim=(1, 2)).sqrt()
                        support_distance = l2_distances.min().item()
                        
                        diag_current_episode["support_distances"].append(support_distance)
                        diag_current_episode["q_values"].append(q_chosen)

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv
                
                if self.diagnostic_enabled and diag_current_episode is not None:
                    diag_current_episode["rewards"].append(reward_venv[self.diagnostic_env_idx])
                    
                    if done_venv[self.diagnostic_env_idx]:
                        if len(diag_current_episode["rewards"]) > 0:
                            diag_current_episode["total_return"] = sum(diag_current_episode["rewards"])
                            if diag_current_episode["q_start"] is not None:
                                diag_current_episode["overestimation"] = (
                                    diag_current_episode["q_start"] - diag_current_episode["total_return"]
                                )
                            if len(diag_current_episode["support_distances"]) > 0:
                                diag_current_episode["avg_support_distance"] = np.mean(diag_current_episode["support_distances"])
                                diag_current_episode["max_support_distance"] = np.max(diag_current_episode["support_distances"])
                        
                        diag_episode_data_this_itr.append(diag_current_episode.copy())
                        diag_current_episode = None

                # add to buffer
                if not eval_mode:
                    obs_venv_copy = obs_venv.copy()
                    for i in range(self.n_envs):
                        if truncated_venv[i]:
                            obs_venv_copy["state"][i] = info_venv[i]["final_obs"][
                                "state"
                            ]
                    obs_buffer.append(prev_obs_venv["state"])
                    next_obs_buffer.append(obs_venv_copy["state"])
                    action_buffer.append(action_venv)
                    reward_buffer.append(reward_venv * self.scale_reward_factor)
                    terminated_buffer.append(terminated_venv)

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            if self.diagnostic_enabled:
                diag_episode_data.extend(diag_episode_data_this_itr)
            
            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                episode_best_reward = np.array(
                    [
                        np.max(reward_traj) / self.act_steps
                        for reward_traj in reward_trajs_split
                    ]
                )
                avg_episode_reward = np.mean(episode_reward)
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                num_batch = int(
                    self.n_steps * self.n_envs / self.batch_size * self.replay_ratio
                )

                obs_trajs = np.array(deepcopy(obs_buffer))
                action_trajs = np.array(deepcopy(action_buffer))
                next_obs_trajs = np.array(deepcopy(next_obs_buffer))
                reward_trajs = np.array(deepcopy(reward_buffer))
                terminated_trajs = np.array(deepcopy(terminated_buffer))

                # flatten
                obs_trajs = einops.rearrange(
                    obs_trajs,
                    "s e h d -> (s e) h d",
                )
                next_obs_trajs = einops.rearrange(
                    next_obs_trajs,
                    "s e h d -> (s e) h d",
                )
                action_trajs = einops.rearrange(
                    action_trajs,
                    "s e h d -> (s e) h d",
                )
                reward_trajs = reward_trajs.reshape(-1)
                terminated_trajs = terminated_trajs.reshape(-1)
                for _ in range(num_batch):

                    # Sample batch
                    inds = np.random.choice(len(obs_trajs), self.batch_size)
                    obs_b = torch.from_numpy(obs_trajs[inds]).float().to(self.device)
                    next_obs_b = (
                        torch.from_numpy(next_obs_trajs[inds]).float().to(self.device)
                    )
                    actions_b = (
                        torch.from_numpy(action_trajs[inds]).float().to(self.device)
                    )
                    reward_b = (
                        torch.from_numpy(reward_trajs[inds]).float().to(self.device)
                    )
                    terminated_b = (
                        torch.from_numpy(terminated_trajs[inds]).float().to(self.device)
                    )

                    # update critic value function
                    critic_loss_v = self.model.loss_critic_v(
                        {"state": obs_b}, actions_b
                    )
                    self.critic_v_optimizer.zero_grad()
                    critic_loss_v.backward()
                    self.critic_v_optimizer.step()

                    # update critic q function
                    critic_loss_q = self.model.loss_critic_q(
                        {"state": obs_b},
                        {"state": next_obs_b},
                        actions_b,
                        reward_b,
                        terminated_b,
                        self.gamma,
                    )
                    self.critic_q_optimizer.zero_grad()
                    critic_loss_q.backward()
                    self.critic_q_optimizer.step()

                    # update target q function
                    self.model.update_target_critic(self.critic_tau)
                    loss_critic = critic_loss_q.detach() + critic_loss_v.detach()

                    # Update policy with collected trajectories - no weighting
                    loss_actor = self.model.loss(
                        actions_b,
                        {"state": obs_b},
                    )
                    self.actor_optimizer.zero_grad()
                    loss_actor.backward()
                    if self.itr >= self.n_critic_warmup_itr:
                        if self.max_grad_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.actor.parameters(), self.max_grad_norm
                            )
                        self.actor_optimizer.step()

            # Update lr
            self.actor_lr_scheduler.step()
            self.critic_v_lr_scheduler.step()
            self.critic_q_lr_scheduler.step()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        log_dict = {
                            "success rate - eval": success_rate,
                            "avg episode reward - eval": avg_episode_reward,
                            "avg best reward - eval": avg_best_reward,
                            "num episode - eval": num_episode_finished,
                        }
                        
                        if self.diagnostic_enabled and len(diag_episode_data) > 0:
                            if len(diag_episode_data_this_itr) > 0:
                                latest_ep = diag_episode_data_this_itr[-1]
                            else:
                                latest_ep = diag_episode_data[-1] if len(diag_episode_data) > 0 else None
                            
                            if latest_ep is not None and "total_return" in latest_ep:
                                log_dict.update({
                                    "diag/q_start": latest_ep["q_start"] if latest_ep["q_start"] is not None else 0.0,
                                    "diag/return_real": latest_ep["total_return"],
                                    "diag/overestimation": latest_ep.get("overestimation", 0.0),
                                    "diag/avg_support_distance": latest_ep.get("avg_support_distance", 0.0),
                                    "diag/max_support_distance": latest_ep.get("max_support_distance", 0.0),
                                })
                            
                            if len(diag_episode_data) > 0:
                                all_q_vals = []
                                all_support_dists = []
                                for ep in diag_episode_data:
                                    all_q_vals.extend(ep.get("q_values", []))
                                    all_support_dists.extend(ep.get("support_distances", []))
                                
                                if len(all_q_vals) > 0 and len(all_support_dists) > 0:
                                    log_dict.update({
                                        "diag/step_q_mean": np.mean(all_q_vals),
                                        "diag/step_q_std": np.std(all_q_vals),
                                        "diag/step_support_mean": np.mean(all_support_dists),
                                        "diag/step_support_std": np.std(all_support_dists),
                                    })
                        
                        wandb.log(log_dict, step=self.itr, commit=False)
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss actor {loss_actor:8.4f} | reward {avg_episode_reward:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        log_dict = {
                            "total env step": cnt_train_step,
                            "loss - actor": loss_actor,
                            "loss - critic": loss_critic,
                            "avg episode reward - train": avg_episode_reward,
                            "num episode - train": num_episode_finished,
                        }
                        
                        if self.diagnostic_enabled and len(diag_episode_data) > 0:
                            if len(diag_episode_data_this_itr) > 0:
                                latest_ep = diag_episode_data_this_itr[-1]
                            else:
                                latest_ep = diag_episode_data[-1] if len(diag_episode_data) > 0 else None
                            
                            if latest_ep is not None and "total_return" in latest_ep:
                                log_dict.update({
                                    "diag/q_start": latest_ep["q_start"] if latest_ep["q_start"] is not None else 0.0,
                                    "diag/return_real": latest_ep["total_return"],
                                    "diag/overestimation": latest_ep.get("overestimation", 0.0),
                                    "diag/avg_support_distance": latest_ep.get("avg_support_distance", 0.0),
                                    "diag/max_support_distance": latest_ep.get("max_support_distance", 0.0),
                                })
                            
                            if len(diag_episode_data) > 0:
                                all_q_vals = []
                                all_support_dists = []
                                for ep in diag_episode_data:
                                    all_q_vals.extend(ep.get("q_values", []))
                                    all_support_dists.extend(ep.get("support_distances", []))
                                
                                if len(all_q_vals) > 0 and len(all_support_dists) > 0:
                                    log_dict.update({
                                        "diag/step_q_mean": np.mean(all_q_vals),
                                        "diag/step_q_std": np.std(all_q_vals),
                                        "diag/step_support_mean": np.mean(all_support_dists),
                                        "diag/step_support_std": np.std(all_support_dists),
                                    })
                        
                        wandb.log(log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
