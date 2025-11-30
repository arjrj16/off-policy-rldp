"""
Environment wrapper for Push-T (PyMunk) environments with state observations.

The Push-T environment has:
- State: [agent_x, agent_y, block_x, block_y, block_angle] (5D)
- Action: [target_agent_x, target_agent_y] (2D)
- Reward: coverage ratio (0-1) based on T-block alignment with goal

For consistency, we will use Dict{} for the observation space, with the key "state" for the state observation.
"""

import numpy as np
import gym
from gym import spaces
import imageio


class PushTLowdimWrapper(gym.Env):
    def __init__(
        self,
        env,
        normalization_path,
        render_hw=(256, 256),
    ):
        self.env = env
        self.render_hw = render_hw
        self.video_writer = None

        # setup spaces
        self.action_space = env.action_space
        normalization = np.load(normalization_path)
        self.obs_min = normalization["obs_min"]
        self.obs_max = normalization["obs_max"]
        self.action_min = normalization["action_min"]
        self.action_max = normalization["action_max"]

        self.observation_space = spaces.Dict()
        obs_example, _ = self.env.reset()  # Push-T returns (obs, info) tuple
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space["state"] = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype,
        )

    def seed(self, seed=None):
        if seed is not None:
            self.env.seed(seed)
            np.random.seed(seed=seed)
        else:
            np.random.seed()

    def reset(self, options={}, **kwargs):
        """Reset the environment with optional video recording."""
        # Close video if exists
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

        # Start video if specified
        if "video_path" in options:
            self.video_writer = imageio.get_writer(options["video_path"], fps=10)

        # Handle seed
        new_seed = options.get("seed", None)
        if new_seed is not None:
            self.seed(seed=new_seed)
        
        raw_obs, info = self.env.reset()

        # normalize
        obs = self.normalize_obs(raw_obs)
        
        # Record first frame if video is being recorded
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        return {"state": obs}

    def normalize_obs(self, obs):
        return 2 * ((obs - self.obs_min) / (self.obs_max - self.obs_min + 1e-6) - 0.5)

    def unnormalize_action(self, action):
        action = (action + 1) / 2  # [-1, 1] -> [0, 1]
        return action * (self.action_max - self.action_min) + self.action_min

    def step(self, action):
        raw_action = self.unnormalize_action(action)
        # Push-T uses new gym API: (obs, reward, terminated, truncated, info)
        raw_obs, reward, terminated, truncated, info = self.env.step(raw_action)

        # normalize
        obs = self.normalize_obs(raw_obs)

        # Record frame if video is being recorded
        if self.video_writer is not None:
            video_img = self.render(mode="rgb_array")
            self.video_writer.append_data(video_img)

        # Return old gym API format (obs, reward, done, info) for compatibility
        done = terminated or truncated
        return {"state": obs}, reward, done, info

    def render(self, mode="rgb_array"):
        img = self.env.render(mode=mode)
        # Resize if needed
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
