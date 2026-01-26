"""MJX environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from gym import spaces

import jax
import jax.numpy as jnp
import mujoco
import mujoco.mjx as mjx

from dm_control.suite import cheetah


def _load_cheetah_model() -> mjx.Model:
    xml_string, assets = cheetah.get_model_and_assets()
    model = mujoco.MjModel.from_xml_string(xml_string, assets=assets)
    return mjx.put_model(model)


def _broadcast_done(done: jnp.ndarray, target_ndim: int) -> jnp.ndarray:
    shape = (done.shape[0],) + (1,) * (target_ndim - 1)
    return done.reshape(shape)


def _mask_tree(new, old, done: jnp.ndarray):
    def _mask_array(n, o):
        mask = _broadcast_done(done, n.ndim)
        return jnp.where(mask, o, n)

    return jax.tree_util.tree_map(_mask_array, new, old)


@jax.tree_util.register_pytree_node_class
@dataclass
class MJXEnvState:
    data: mjx.Data
    obs_hist: jnp.ndarray
    step_count: jnp.ndarray
    success: jnp.ndarray
    done: jnp.ndarray
    rng: jnp.ndarray

    def tree_flatten(self):
        children = (
            self.data,
            self.obs_hist,
            self.step_count,
            self.success,
            self.done,
            self.rng,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


class MJXCheetahRunVecEnv:
    def __init__(
        self,
        n_envs: int,
        obs_steps: int = 1,
        max_episode_steps: int = 1000,
        reward_threshold: float = 5.0,
        seed: int = 0,
    ):
        self.n_envs = n_envs
        self.num_envs = n_envs
        self.obs_steps = obs_steps
        self.max_episode_steps = max_episode_steps
        self.reward_threshold = reward_threshold
        self.model = _load_cheetah_model()

        # DM Control cheetah uses position without root x
        self.obs_dim = int(self.model.nq - 1 + self.model.nv)
        self.action_dim = int(self.model.nu)

        self.single_observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.obs_steps, self.obs_dim),
                    dtype=np.float32,
                )
            }
        )
        self.single_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.action_dim,),
            dtype=np.float32,
        )
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        self.metadata = {
            "render.modes": [],
            "video.frames_per_second": 0,
        }

        model = self.model
        obs_steps = self.obs_steps
        max_steps = self.max_episode_steps
        reward_threshold = self.reward_threshold

        def reset_single(key):
            data = mjx.make_data(model)
            noise = jax.random.normal(key, (model.nq + model.nv,)) * 0.01
            qpos = model.qpos0 + noise[: model.nq]
            qvel = noise[model.nq :]
            return data.replace(qpos=qpos, qvel=qvel)

        def get_obs(data: mjx.Data) -> jnp.ndarray:
            qpos = data.qpos[:, 1:]
            qvel = data.qvel
            return jnp.concatenate([qpos, qvel], axis=-1)

        def reset_fn(rng: jnp.ndarray) -> Tuple[MJXEnvState, jnp.ndarray]:
            rng, subkey = jax.random.split(rng)
            keys = jax.random.split(subkey, n_envs)
            data = jax.vmap(reset_single)(keys)
            obs = get_obs(data)
            obs_hist = jnp.repeat(obs[:, None, :], obs_steps, axis=1)
            step_count = jnp.zeros((n_envs,), dtype=jnp.int32)
            success = jnp.zeros((n_envs,), dtype=jnp.bool_)
            done = jnp.zeros((n_envs,), dtype=jnp.bool_)
            state = MJXEnvState(
                data=data,
                obs_hist=obs_hist,
                step_count=step_count,
                success=success,
                done=done,
                rng=rng,
            )
            return state, obs_hist

        def step_fn(
            state: MJXEnvState, actions: jnp.ndarray
        ) -> Tuple[MJXEnvState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            if actions.ndim == 2:
                actions = actions[:, None, :]
            actions = jnp.clip(actions, -1.0, 1.0)
            actions = jnp.swapaxes(actions, 0, 1)

            def body(carry, act):
                data, step_count, success, done, obs_hist = carry
                done_prev = done
                def step_single(d, a):
                    return mjx.step(model, d.replace(ctrl=a))

                data_next = jax.vmap(step_single)(data, act)
                vel = data_next.qvel[:, 0]
                success_next = success | (vel >= reward_threshold)
                step_count_next = step_count + 1
                done_next = step_count_next >= max_steps

                data = _mask_tree(data_next, data, done)
                step_count = jnp.where(done, step_count, step_count_next)
                success = jnp.where(done, success, success_next)
                reward = jnp.where(
                    (~done) & done_next, success_next.astype(jnp.float32), 0.0
                )
                done = done | done_next
                obs = get_obs(data)
                if obs_steps > 1:
                    obs_hist_next = jnp.concatenate(
                        [obs_hist[:, 1:, :], obs[:, None, :]], axis=1
                    )
                else:
                    obs_hist_next = obs[:, None, :]
                obs_hist = jnp.where(
                    _broadcast_done(done_prev, obs_hist_next.ndim),
                    obs_hist,
                    obs_hist_next,
                )
                return (data, step_count, success, done, obs_hist), reward

            (data, step_count, success, done, obs_hist), rewards = jax.lax.scan(
                body,
                (state.data, state.step_count, state.success, state.done, state.obs_hist),
                actions,
            )
            reward = rewards.sum(axis=0)
            new_state = MJXEnvState(
                data=data,
                obs_hist=obs_hist,
                step_count=step_count,
                success=success,
                done=done,
                rng=state.rng,
            )
            return new_state, obs_hist, reward, done

        self._reset_fn = jax.jit(reset_fn)
        self._step_fn = jax.jit(step_fn)
        self._reset_single = reset_single
        self._get_obs = get_obs

        self.seed(seed)
        self.reset()

    def seed(self, seed: Optional[int] = None):
        if seed is None:
            seed = 0
        if isinstance(seed, (list, tuple, np.ndarray)):
            seed = int(seed[0])
        self._rng = jax.random.PRNGKey(int(seed))

    def reset(self, **kwargs) -> Dict[str, np.ndarray]:
        self.state, obs_hist = self._reset_fn(self._rng)
        self._rng = self.state.rng
        return {"state": np.array(jax.device_get(obs_hist))}

    def reset_arg(self, options_list=None, **kwargs) -> Dict[str, np.ndarray]:
        return self.reset()

    def reset_one_arg(self, env_ind: int, options=None):
        key, new_key = jax.random.split(self._rng)
        data_single = self._reset_single(key)
        data = jax.tree_util.tree_map(
            lambda x, y: x.at[env_ind].set(y),
            self.state.data,
            data_single,
        )
        obs = self._get_obs(data)
        if self.obs_steps > 1:
            obs_hist = self.state.obs_hist.at[env_ind].set(
                jnp.repeat(obs[env_ind][None, :], self.obs_steps, axis=0)
            )
        else:
            obs_hist = self.state.obs_hist.at[env_ind].set(obs[env_ind][None, :])
        step_count = self.state.step_count.at[env_ind].set(0)
        success = self.state.success.at[env_ind].set(False)
        done = self.state.done.at[env_ind].set(False)
        self.state = MJXEnvState(
            data=data,
            obs_hist=obs_hist,
            step_count=step_count,
            success=success,
            done=done,
            rng=self.state.rng,
        )
        self._rng = new_key
        return {"state": np.array(jax.device_get(obs_hist[env_ind]))}

    def step(self, actions: np.ndarray):
        actions = jnp.asarray(actions)
        self.state, obs_hist, reward, done = self._step_fn(self.state, actions)
        obs = {"state": np.array(jax.device_get(obs_hist))}
        reward = np.array(jax.device_get(reward))
        done = np.array(jax.device_get(done))
        terminated = np.zeros_like(done, dtype=bool)
        truncated = done
        info = [{} for _ in range(self.n_envs)]
        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        return None

    def close(self):
        return None


__all__ = ["MJXCheetahRunVecEnv"]
