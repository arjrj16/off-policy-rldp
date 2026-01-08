"""
Conditional flow matching (CFM) backbone for action-sequence policies.

We model a velocity field v_theta(x, t | cond) and train it with a simple
linear-interpolation flow matching objective:

  x0 ~ N(0, I)
  x1 = action trajectory from data
  t ~ Uniform(0, 1)
  x_t = (1 - t) * x0 + t * x1
  v*(x_t, t) = x1 - x0
  L = E[ || v_theta(x_t, t, cond) - v*(x_t, t) ||^2 ]

Sampling integrates the ODE dx/dt = v_theta(x, t, cond) from t=0 -> 1.
"""

import logging
import torch
from torch import nn
from typing import Optional, Union

log = logging.getLogger(__name__)


class FlowMatchingModel(nn.Module):
    def __init__(
        self,
        network: nn.Module,
        horizon_steps: int,
        obs_dim: int,
        action_dim: int,
        network_path: Optional[str] = None,
        device: str = "cuda:0",
        # Sampling (ODE integration)
        flow_steps: int = 20,
        sampler: str = "euler",  # "euler" | "heun"
        init_noise_scale: float = 1.0,
        randn_clip_value: float = 10.0,
        final_action_clip_value: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        self.device = device
        self.horizon_steps = int(horizon_steps)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)

        self.network = network.to(device)
        if network_path is not None:
            # Match DiffusionModel semantics: if checkpoint has "ema", load that.
            checkpoint = torch.load(network_path, map_location=device)
            if isinstance(checkpoint, dict) and ("ema" in checkpoint):
                self.load_state_dict(checkpoint["ema"], strict=False)
                log.info("Loaded SL-trained policy from %s", network_path)
            else:
                # support older checkpoints that store "model" only
                if isinstance(checkpoint, dict) and ("model" in checkpoint):
                    self.load_state_dict(checkpoint["model"], strict=False)
                else:
                    # raw state_dict
                    self.load_state_dict(checkpoint, strict=False)
                log.info("Loaded policy from %s", network_path)

        self.flow_steps = int(flow_steps)
        assert self.flow_steps > 0, "flow_steps must be > 0"
        self.sampler = str(sampler).lower()
        assert self.sampler in {"euler", "heun"}, f"Unknown sampler: {sampler}"

        self.init_noise_scale = float(init_noise_scale)
        self.randn_clip_value = float(randn_clip_value)
        self.final_action_clip_value = final_action_clip_value

    def _make_t(self, batch_size: int, t: Union[float, torch.Tensor], device: torch.device):
        if torch.is_tensor(t):
            t_b = t.to(device=device, dtype=torch.float32).view(batch_size)
        else:
            t_b = torch.full((batch_size,), float(t), device=device, dtype=torch.float32)
        return t_b

    @torch.no_grad()
    def sample(
        self,
        cond: dict,
        deterministic: bool = False,
        network_override: Optional[nn.Module] = None,
    ):
        """
        Sample action trajectories by integrating dx/dt = v_theta(x, t, cond).

        Args:
            cond: dict with "state" key, shape (B, To, Do)
        Returns:
            actions: (B, horizon_steps, action_dim)
        """
        # NOTE: "deterministic" currently matches repo semantics: no extra noise is
        # injected during integration, but diversity still comes from x0 ~ N(0, I).
        _ = deterministic

        device = torch.device(self.device)
        B = len(cond["state"])
        x = torch.randn((B, self.horizon_steps, self.action_dim), device=device) * self.init_noise_scale
        x = x.clamp(-self.randn_clip_value, self.randn_clip_value)

        dt = 1.0 / float(self.flow_steps)
        net = network_override if network_override is not None else self.network

        for i in range(self.flow_steps):
            t = float(i) * dt
            t_b = self._make_t(B, t, device)
            v = net(x, t_b, cond=cond)
            if self.sampler == "euler":
                x = x + dt * v
            else:  # heun (RK2)
                x_euler = x + dt * v
                t_b_next = self._make_t(B, t + dt, device)
                v_next = net(x_euler, t_b_next, cond=cond)
                x = x + 0.5 * dt * (v + v_next)

        if self.final_action_clip_value is not None:
            x = torch.clamp(x, -self.final_action_clip_value, self.final_action_clip_value)
        return x

    @torch.no_grad()
    def forward(self, cond: dict, deterministic: bool = False):
        return self.sample(cond=cond, deterministic=deterministic)

    def loss(self, x1: torch.Tensor, cond: dict):
        """
        Flow matching training loss.

        Args:
            x1: (B, horizon_steps, action_dim) target action trajectory
            cond: dict with "state" key
        """
        device = x1.device
        B = len(x1)

        x0 = torch.randn_like(x1, device=device).clamp(-self.randn_clip_value, self.randn_clip_value)
        t = torch.rand((B,), device=device, dtype=torch.float32)
        t_view = t.view(B, 1, 1)

        x_t = (1.0 - t_view) * x0 + t_view * x1
        v_target = x1 - x0
        v_pred = self.network(x_t, t, cond=cond)
        return torch.mean((v_pred - v_target) ** 2)

