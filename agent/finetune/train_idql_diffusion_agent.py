"""
Implicit diffusion Q-learning (IDQL) trainer for diffusion policy.

Do not support pixel input right now.

"""

import os
import pickle
import csv
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

        # BC warm-up for sparse reward tasks
        self.use_bc_warmup = cfg.train.get("use_bc_warmup", False)
        self.bc_warmup_iters = cfg.train.get("bc_warmup_iters", 0)
        if self.use_bc_warmup:
            log.info(f"BC warm-up enabled: using BC policy without Q-filtering for first {self.bc_warmup_iters} iterations")

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
            # Initialize CSV file for step-level diagnostics
            self.diagnostic_csv_path = os.path.join(self.logdir, "diagnostics.csv")
            self._diag_csv_f = open(self.diagnostic_csv_path, "w", newline="")
            csv_fieldnames = [
                "itr", "global_env_step",
                "episode_id", "t_in_episode",
                "reward_scaled", "rtg",
                "q_exec", "q_argmax",
                "support_exec", "support_argmax",
                "oe_exec", "oe_argmax",
                "eval_mode", "task"
            ]
            self._diag_csv = csv.DictWriter(self._diag_csv_f, fieldnames=csv_fieldnames)
            self._diag_csv.writeheader()
            self._diag_csv_f.flush()
            self._diag_episode_counter = 0
        
        # Argmax-Q trajectory visualization (for IDQL stability plots)
        argmax_cfg = cfg.train.get("argmax_traj", {})
        # Turn this on to dump argmax-Q rollouts to npz
        self.argmax_traj_enabled = argmax_cfg.get("enabled", False)
        # Collect at itr == 0 and then every `freq` itrs if > 0
        self.argmax_traj_freq = argmax_cfg.get("freq", 0)
        # How many environment steps to roll out when collecting trajectories
        self.argmax_traj_steps = argmax_cfg.get("steps", self.n_steps)
        # How many candidate actions to sample per state when searching for argmax-Q
        self.argmax_traj_num_sample = argmax_cfg.get("num_sample", self.num_sample)

    @torch.no_grad()
    def rollout_argmax_trajectories(self, itr_tag: int) -> None:
        """
        Roll out the current IDQL policy using, at each state, the action
        sample with maximal Q-value. Saves states (and per-env Q entropies)
        to an .npz file in self.logdir so you can make DPPO-style plots.

        The resulting file has:
            states:   [T, n_env, H, D]
            q_entropy:[T, n_env]    (Shannon entropy over the S Q-samples)
        """
        if not getattr(self, "venv", None):
            log.warning("No vector env attached; cannot collect argmax trajectories.")
            return

        log.info(
            f"[ArgmaxTraj] Collecting argmax-Q trajectories at itr={itr_tag} "
            f"for {self.argmax_traj_steps} steps"
        )

        self.model.eval()

        # Fresh reset for all envs so this doesn't depend on training rollouts
        options_venv = [{'seed': 42 + i} for i in range(self.n_envs)]
        obs_venv = self.reset_env_all(options_venv=options_venv)

        states_over_time = []
        entropies_over_time = []
        actions_over_time = []

        for step in range(self.argmax_traj_steps):
            # Record current states
            # obs_venv["state"] has shape (n_env, H, D)
            states_over_time.append(obs_venv["state"].copy())

            cond = {
                "state": torch.from_numpy(obs_venv["state"])
                .float()
                .to(self.device)
            }

            # We want *stochastic* samples in order to search over Q, so deterministic=False
            samples, diag = self.model(
                cond=cond,
                deterministic=False,
                num_sample=self.argmax_traj_num_sample,
                use_expectile_exploration=False,
                use_bc_warmup=False,
                return_diagnostics=True,
            )

            # diag["all_q"]: (S, B), diag["all_samples"]: (S, B, H, A)
            all_q = diag["all_q"]            # [S, n_env]
            all_samples = diag["all_samples"]  # [S, n_env, H, A]

            # Argmax index per env along the sample dimension
            argmax_idx = torch.argmax(all_q, dim=0)  # [n_env]

            # Also compute per-env entropy over Q-samples as a diffuseness measure
            entropies = []
            for env_idx in range(self.n_envs):
                q_col = all_q[:, env_idx]
                probs = torch.softmax(q_col, dim=0)
                entropy = -(probs * torch.log(probs + 1e-8)).sum().item()
                entropies.append(entropy)
            entropies_over_time.append(entropies)

            # Build action tensor where each env takes its argmax-Q sample
            action_list = []
            for env_idx in range(self.n_envs):
                idx = argmax_idx[env_idx]
                a = all_samples[idx, env_idx, : self.act_steps, :]  # (H, A)
                action_list.append(a.unsqueeze(0))
            action_venv = torch.cat(action_list, dim=0).cpu().numpy()  # (n_env, H, A)

            actions_over_time.append(action_venv.copy())

            # Step vector env; we rely on gym's auto-reset-on-done semantics
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = self.venv.step(
                action_venv
            )

        # Stack and save
        states_arr = np.stack(states_over_time, axis=0)      # [T, n_env, H, D]
        entropy_arr = np.stack(entropies_over_time, axis=0)  # [T, n_env]
        actions_arr = np.stack(actions_over_time, axis=0)

        save_path = os.path.join(self.logdir, f"argmax_traj_itr{itr_tag:06d}.npz")
        # np.savez_compressed(save_path, states=states_arr, q_entropy=entropy_arr)
        np.savez_compressed(
            save_path,
            states=states_arr,
            q_entropy=entropy_arr,
            actions=actions_arr,
        )
        log.info(f"[ArgmaxTraj] Saved argmax-Q trajectories to {save_path}")

        self.model.train()


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
                    
                    # Determine if we should use BC warm-up
                    use_bc_warmup_this_step = self.use_bc_warmup and (self.itr < self.bc_warmup_iters)
                    
                    # Get action with diagnostics if enabled
                    if self.diagnostic_enabled:
                        result = self.model(
                            cond=cond,
                            deterministic=eval_mode and self.eval_deterministic,
                            num_sample=self.num_sample,
                            use_expectile_exploration=self.use_expectile_exploration,
                            return_diagnostics=True,
                            use_bc_warmup=use_bc_warmup_this_step,
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
                                use_bc_warmup=use_bc_warmup_this_step,
                            )
                            .cpu()
                            .numpy()
                        )  # n_env x horizon x act
                        diag = None
                    
                action_venv = samples[:, : self.act_steps]
                
                # Compute diagnostics for tracked environment
                if self.diagnostic_enabled and diag is not None and self.diagnostic_env_idx < self.n_envs:
                    # Check if episode is still active (not done from previous step)
                    # BUT: if this is a new episode start, env_was_done should be False
                    env_was_done = done_venv[self.diagnostic_env_idx] if step > 0 else False
                    
                    if firsts_trajs[step, self.diagnostic_env_idx] == 1:
                        if diag_current_episode is not None:
                            log.warning(f"Starting new episode but diag_current_episode exists at step {step}, itr {self.itr}")
                            diag_current_episode = None
                        diag_current_episode = {
                            "q_start": None,
                            "rewards": [],
                            "q_exec": [],
                            "q_argmax": [],
                            "support_exec": [],
                            "support_argmax": [],
                            "step": step,
                            "itr": self.itr,
                            "episode_id": self._diag_episode_counter,
                        }
                        self._diag_episode_counter += 1
                        # Reset env_was_done for new episode - we want to compute Q/support for the first step
                        env_was_done = False
                    
                    # Compute Q-value for chosen action (for env 0) - only if episode is active
                    if diag_current_episode is not None and not env_was_done:
                        # Get Q-values: both executed and argmax
                        chosen_idx = diag["chosen_indices"][self.diagnostic_env_idx].item()
                        q_exec = diag["all_q"][chosen_idx, self.diagnostic_env_idx].item()
                        
                        # Get argmax Q-value
                        q_col = diag["all_q"][:, self.diagnostic_env_idx]  # (S,)
                        argmax_idx = int(torch.argmax(q_col).item())
                        q_argmax = float(q_col[argmax_idx].item())
                        
                        # Store Q at episode start (use executed Q)
                        if diag_current_episode["q_start"] is None:
                            diag_current_episode["q_start"] = q_exec
                        
                        # Sample from frozen BC to compute support distance
                        cond_env0 = {
                            "state": cond["state"][self.diagnostic_env_idx:self.diagnostic_env_idx+1]  # (1, T, D)
                        }
                        bc_samples = self.model.sample_from_frozen_bc(
                            cond_env0,
                            num_samples=self.diagnostic_bc_num_samples,
                            deterministic=False,
                        )  # (num_samples, 1, H, A)
                        bc = bc_samples[:, 0, :self.act_steps, :]  # (K, act_steps, A)
                        
                        # Get actions from all_samples (torch tensor) for consistency
                        all_samples = diag["all_samples"]  # (S, B, H, A)
                        a_exec = all_samples[chosen_idx, self.diagnostic_env_idx, :self.act_steps, :]  # (act_steps, A)
                        a_argmax = all_samples[argmax_idx, self.diagnostic_env_idx, :self.act_steps, :]  # (act_steps, A)
                        
                        # Compute support distance for both actions
                        def min_l2_to_bc(a):  # a: (act_steps, A)
                            diff = bc - a.unsqueeze(0)  # (K, act_steps, A)
                            d2 = (diff ** 2).sum(dim=(1, 2))
                            return float(torch.sqrt(d2.min()).item())
                        
                        support_exec = min_l2_to_bc(a_exec)
                        support_argmax = min_l2_to_bc(a_argmax)
                        
                        diag_current_episode["q_exec"].append(q_exec)
                        diag_current_episode["q_argmax"].append(q_argmax)
                        diag_current_episode["support_exec"].append(support_exec)
                        diag_current_episode["support_argmax"].append(support_argmax)

                # Apply multi-step action
                obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                    self.venv.step(action_venv)
                )
                done_venv = terminated_venv | truncated_venv
                reward_trajs[step] = reward_venv
                firsts_trajs[step + 1] = done_venv
                
                if self.diagnostic_enabled and diag_current_episode is not None:
                    # Store scaled reward
                    r = reward_venv[self.diagnostic_env_idx] * self.scale_reward_factor
                    diag_current_episode["rewards"].append(float(r))
                    
                    if done_venv[self.diagnostic_env_idx]:
                        if len(diag_current_episode["rewards"]) > 0:
                            # Compute discounted return (scaled rewards already stored)
                            rewards = diag_current_episode["rewards"]  # already scaled
                            G0 = 0.0
                            g = 1.0
                            for r in rewards:
                                G0 += g * r
                                g *= self.gamma
                            
                            diag_current_episode["return_real"] = G0  # discounted, scaled
                            if diag_current_episode["q_start"] is not None:
                                diag_current_episode["overestimation"] = diag_current_episode["q_start"] - G0
                            
                            # Compute per-step return-to-go
                            rtg = [0.0] * len(rewards)
                            running = 0.0
                            for t in reversed(range(len(rewards))):
                                running = rewards[t] + self.gamma * running
                                rtg[t] = running
                            diag_current_episode["rtg"] = rtg
                            
                            # Compute support distance stats
                            if len(diag_current_episode["support_exec"]) > 0:
                                diag_current_episode["avg_support_exec"] = np.mean(diag_current_episode["support_exec"])
                                diag_current_episode["max_support_exec"] = np.max(diag_current_episode["support_exec"])
                                diag_current_episode["avg_support_argmax"] = np.mean(diag_current_episode["support_argmax"])
                                diag_current_episode["max_support_argmax"] = np.max(diag_current_episode["support_argmax"])
                            
                            # Write CSV rows for this episode
                            episode_id = diag_current_episode["episode_id"]
                            itr = diag_current_episode["itr"]
                            task = getattr(self, "env_name", "unknown")
                            
                            for t in range(len(rewards)):
                                oe_exec = diag_current_episode["q_exec"][t] - rtg[t] if t < len(diag_current_episode["q_exec"]) else 0.0
                                oe_argmax = diag_current_episode["q_argmax"][t] - rtg[t] if t < len(diag_current_episode["q_argmax"]) else 0.0
                                
                                row = {
                                    "itr": itr,
                                    "global_env_step": cnt_train_step,
                                    "episode_id": episode_id,
                                    "t_in_episode": t,
                                    "reward_scaled": rewards[t],
                                    "rtg": rtg[t],
                                    "q_exec": diag_current_episode["q_exec"][t] if t < len(diag_current_episode["q_exec"]) else 0.0,
                                    "q_argmax": diag_current_episode["q_argmax"][t] if t < len(diag_current_episode["q_argmax"]) else 0.0,
                                    "support_exec": diag_current_episode["support_exec"][t] if t < len(diag_current_episode["support_exec"]) else 0.0,
                                    "support_argmax": diag_current_episode["support_argmax"][t] if t < len(diag_current_episode["support_argmax"]) else 0.0,
                                    "oe_exec": oe_exec,
                                    "oe_argmax": oe_argmax,
                                    "eval_mode": 1 if eval_mode else 0,
                                    "task": task,
                                }
                                self._diag_csv.writerow(row)
                            self._diag_csv_f.flush()
                        
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
                            
                            if latest_ep is not None and "return_real" in latest_ep:
                                log_dict.update({
                                    "diag/q_start": latest_ep["q_start"] if latest_ep["q_start"] is not None else 0.0,
                                    "diag/return_real": latest_ep["return_real"],
                                    "diag/overestimation": latest_ep.get("overestimation", 0.0),
                                    "diag/avg_support_exec": latest_ep.get("avg_support_exec", 0.0),
                                    "diag/max_support_exec": latest_ep.get("max_support_exec", 0.0),
                                    "diag/avg_support_argmax": latest_ep.get("avg_support_argmax", 0.0),
                                    "diag/max_support_argmax": latest_ep.get("max_support_argmax", 0.0),
                                })
                            
                            if len(diag_episode_data) > 0:
                                all_q_exec = []
                                all_q_argmax = []
                                all_support_exec = []
                                all_support_argmax = []
                                for ep in diag_episode_data:
                                    all_q_exec.extend(ep.get("q_exec", []))
                                    all_q_argmax.extend(ep.get("q_argmax", []))
                                    all_support_exec.extend(ep.get("support_exec", []))
                                    all_support_argmax.extend(ep.get("support_argmax", []))
                                
                                if len(all_q_exec) > 0:
                                    log_dict.update({
                                        "diag/step_q_exec_mean": np.mean(all_q_exec),
                                        "diag/step_q_exec_std": np.std(all_q_exec),
                                        "diag/step_q_argmax_mean": np.mean(all_q_argmax),
                                        "diag/step_q_argmax_std": np.std(all_q_argmax),
                                    })
                                if len(all_support_exec) > 0:
                                    log_dict.update({
                                        "diag/step_support_exec_mean": np.mean(all_support_exec),
                                        "diag/step_support_exec_std": np.std(all_support_exec),
                                        "diag/step_support_argmax_mean": np.mean(all_support_argmax),
                                        "diag/step_support_argmax_std": np.std(all_support_argmax),
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
                            
                            if latest_ep is not None and "return_real" in latest_ep:
                                log_dict.update({
                                    "diag/q_start": latest_ep["q_start"] if latest_ep["q_start"] is not None else 0.0,
                                    "diag/return_real": latest_ep["return_real"],
                                    "diag/overestimation": latest_ep.get("overestimation", 0.0),
                                    "diag/avg_support_exec": latest_ep.get("avg_support_exec", 0.0),
                                    "diag/max_support_exec": latest_ep.get("max_support_exec", 0.0),
                                    "diag/avg_support_argmax": latest_ep.get("avg_support_argmax", 0.0),
                                    "diag/max_support_argmax": latest_ep.get("max_support_argmax", 0.0),
                                })
                            
                            if len(diag_episode_data) > 0:
                                all_q_exec = []
                                all_q_argmax = []
                                all_support_exec = []
                                all_support_argmax = []
                                for ep in diag_episode_data:
                                    all_q_exec.extend(ep.get("q_exec", []))
                                    all_q_argmax.extend(ep.get("q_argmax", []))
                                    all_support_exec.extend(ep.get("support_exec", []))
                                    all_support_argmax.extend(ep.get("support_argmax", []))
                                
                                if len(all_q_exec) > 0:
                                    log_dict.update({
                                        "diag/step_q_exec_mean": np.mean(all_q_exec),
                                        "diag/step_q_exec_std": np.std(all_q_exec),
                                        "diag/step_q_argmax_mean": np.mean(all_q_argmax),
                                        "diag/step_q_argmax_std": np.std(all_q_argmax),
                                    })
                                if len(all_support_exec) > 0:
                                    log_dict.update({
                                        "diag/step_support_exec_mean": np.mean(all_support_exec),
                                        "diag/step_support_exec_std": np.std(all_support_exec),
                                        "diag/step_support_argmax_mean": np.mean(all_support_argmax),
                                        "diag/step_support_argmax_std": np.std(all_support_argmax),
                                    })
                        
                        wandb.log(log_dict, step=self.itr, commit=True)
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)

                if self.argmax_traj_enabled:
                    do_collect = (self.itr == 0)
                    if self.argmax_traj_freq > 0 and (self.itr % self.argmax_traj_freq == 0):
                        do_collect = True
                    if do_collect:
                        self.rollout_argmax_trajectories(itr_tag=self.itr)

            self.itr += 1
