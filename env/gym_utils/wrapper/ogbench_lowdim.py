"""
Environment wrapper for OGBench environments with state observations.

Bridges the Gymnasium API (used by OGBench) to the old gym API used by
this codebase.  Applies min-max normalization so observations and actions
live in [-1, 1].

For consistency, we use Dict{} for the observation space, with the key
"state" for the state observation.
"""

import numpy as np
import gym
from gym import spaces


class OGBenchLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
        task_id=None,
        render_hw=(256, 256),
    ):
        self.env = env
        self.task_id = task_id
        self.render_hw = render_hw
        self.video_writer = None

        # Seed to pass to the underlying gymnasium env on the NEXT reset only.
        # Previously seed() seeded just the worker-global numpy RNG, which the
        # OGBench reset path never reads (it uses self.np_random), so initial
        # states were not reproducible from cfg.seed.
        self._pending_seed = None

        # Load normalization stats produced by process_ogbench_dataset.py
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"].astype(np.float32)
        self.obs_max = normalization["obs_max"].astype(np.float32)
        self.action_min = normalization["action_min"].astype(np.float32)
        self.action_max = normalization["action_max"].astype(np.float32)

        # Build action space in normalized [-1, 1] range
        act_dim = env.action_space.shape[0]
        self.action_space = gym.spaces.Box(
            low=-np.ones(act_dim, dtype=np.float32),
            high=np.ones(act_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # Build observation space as Dict{"state": Box(-1, 1)}
        obs_dim = env.observation_space.shape[0]
        self.observation_space = spaces.Dict()
        self.observation_space["state"] = spaces.Box(
            low=-np.ones(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def normalize_obs(self, obs):
        obs = obs.astype(np.float32)
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def seed(self, seed=None):
        # Defer to the next reset: gymnasium only re-derives env.np_random when
        # a seed is passed to reset(), and it must be passed exactly once or
        # every episode would replay the same initial state.
        self._pending_seed = seed
        if seed is not None:
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options=None, **kwargs):
        if options is None:
            options = {}

        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            import imageio
            self.video_writer = imageio.get_writer(options["video_path"], fps=30)

        # Build gymnasium reset options
        gym_options = {}
        if self.task_id is not None:
            gym_options["task_id"] = self.task_id

        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)

        # Apply a pending seed exactly once so the 50 vectorized envs are
        # reproducible (seed+i each) yet evolve independently afterwards.
        reset_kwargs = {"options": gym_options if gym_options else None}
        if self._pending_seed is not None:
            reset_kwargs["seed"] = self._pending_seed
            self._pending_seed = None
        obs, info = self.env.reset(**reset_kwargs)
        obs = self.normalize_obs(obs)

        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            if video_img is not None:
                self.video_writer.append_data(video_img)

        return {"state": obs}

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        obs, reward, terminated, truncated, info = self.env.step(raw_action)
        obs = self.normalize_obs(obs)

        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            if video_img is not None:
                self.video_writer.append_data(video_img)

        done = terminated or truncated
        # Preserve the terminated/truncated distinction across the old-gym
        # 4-tuple API using the convention MultiStep already decodes: set
        # "TimeLimit.truncated" only on a pure timeout (mirrors old gym's
        # TimeLimit wrapper, so MultiStep's own step counter stays active for
        # the no-flag case). Without this, gymnasium TimeLimit timeouts arrive
        # as plain done=True and MultiStep records them as terminations, so the
        # critic skips bootstrapping at timeouts and final_obs is never saved.
        # Termination (success) takes precedence if both fire on the same step.
        if truncated and not terminated:
            info = dict(info)
            info["TimeLimit.truncated"] = True
        return {"state": obs}, reward, done, info

    def render(self, mode="rgb_array"):
        img = self.env.render()
        if img is not None and self.render_hw is not None:
            import cv2
            h, w = self.render_hw
            img = cv2.resize(img, (w, h))
        return img

    def close(self):
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None
        return self.env.close()
