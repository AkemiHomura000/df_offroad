"""GRPO utilities for Verti: PDMS → VertiPDM adapter + diffusion logprob scheduler.

get_verti_score_para is the drop-in for diffusiondrivev2_model_rl.get_pdm_score_para.
Official RL model cannot import in the lean ddv2 env (needs full nuplan/PDMS);
this module reimplements the scoring call + truncated GRPO advantage math.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from diffusers.schedulers import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor

ROOT = Path("/home/rm/Desktop/df_offroad")
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))

from verti_metric_cache import VertiMetricCache, WorldMapBank  # noqa: E402
from verti_pdm_scorer import (  # noqa: E402
    VertiPDMConfig,
    compute_grpo_advantages,
    get_verti_score_para,
    score_batch,
)

__all__ = [
    "DDIMScheduler_with_logprob",
    "bezier_xyyaw",
    "get_verti_score_para",
    "compute_grpo_advantages",
    "score_batch",
    "VertiPDMConfig",
    "WorldMapBank",
    "VertiMetricCache",
    "grpo_advantages_torch",
]


class DDIMScheduler_with_logprob(DDIMScheduler):
    """Copied from diffusiondrivev2_model_rl (scale-adaptive multiplicative noise)."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 1.0,
        use_clipped_model_output: bool = False,
        generator=None,
        variance_noise: Optional[torch.Tensor] = None,
        prev_sample: Optional[torch.FloatTensor] = None,
        return_dict: bool = True,
    ):
        if self.num_inference_steps is None:
            raise ValueError("run set_timesteps first")

        prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = (
            self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        )
        beta_prod_t = 1 - alpha_prod_t

        if self.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
            pred_epsilon = model_output
        elif self.config.prediction_type == "sample":
            pred_original_sample = model_output
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5
        elif self.config.prediction_type == "v_prediction":
            pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
            pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
        else:
            raise ValueError(self.config.prediction_type)

        if self.config.thresholding:
            pred_original_sample = self._threshold_sample(pred_original_sample)
        elif self.config.clip_sample:
            pred_original_sample = pred_original_sample.clamp(
                -self.config.clip_sample_range, self.config.clip_sample_range
            )

        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = (eta * variance ** 0.5).clamp_(min=1e-10)

        if use_clipped_model_output:
            pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

        pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2).clamp_(min=0) ** 0.5 * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if eta > 0:
            std_dev_t_mul = torch.clip(std_dev_t, min=0.04)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)
        else:
            std_dev_t_mul = torch.tensor(0.0).to(std_dev_t.device)
            std_dev_t_add = torch.tensor(0.0).to(std_dev_t.device)

        if prev_sample is None:
            variance_noise_horizon = (
                randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1],
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                * std_dev_t_mul
                + 1.0
            )
            variance_noise_vert = (
                randn_tensor(
                    [model_output.shape[0], model_output.shape[1], 1, 1],
                    generator=generator,
                    device=model_output.device,
                    dtype=model_output.dtype,
                )
                * std_dev_t_mul
                + 1.0
            )
            variance_noise_mul = torch.cat((variance_noise_horizon, variance_noise_vert), dim=-1)
            variance_noise_mul = variance_noise_mul.repeat(1, 1, model_output.shape[2], 1)
            variance_noise_x = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance_noise_y = randn_tensor(
                [model_output.shape[0], model_output.shape[1], 1, 1],
                generator=generator,
                device=model_output.device,
                dtype=model_output.dtype,
            )
            variance_noise_add = torch.cat((variance_noise_x, variance_noise_y), dim=-1)
            variance_noise_add = variance_noise_add.repeat(1, 1, model_output.shape[2], 1)
            prev_sample = prev_sample_mean * variance_noise_mul + std_dev_t_add * variance_noise_add

        std_dev_t_mul = torch.clip(std_dev_t, min=0.1)
        log_prob = (
            -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t_mul ** 2))
            - torch.log(std_dev_t_mul)
            - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
        )
        log_prob = log_prob.sum(dim=(-2, -1))
        return prev_sample.type(sample.dtype), log_prob, prev_sample_mean.type(sample.dtype)


def bezier_xyyaw(xy8: torch.Tensor) -> torch.Tensor:
    assert xy8.shape[-2:] == (8, 2), "Input must be (B,G,8,2)"
    B, G, _, _ = xy8.shape
    device, dtype = xy8.device, xy8.dtype
    origin = torch.zeros_like(xy8[..., :1, :])
    ctrl = torch.cat([origin, xy8], dim=-2)
    n = ctrl.shape[-2] - 1
    delta = ctrl[..., 1:, :] - ctrl[..., :-1, :]
    binom = torch.tensor([math.comb(n - 1, i) for i in range(n)], device=device, dtype=dtype)
    t = torch.arange(1, n + 1, device=device, dtype=dtype) / n
    t_pow = t.view(-1, 1) ** torch.arange(0, n, device=device, dtype=dtype)
    one_pow = (1 - t).view(-1, 1) ** torch.arange(n - 1, -1, -1, device=device, dtype=dtype)
    basis = binom * t_pow * one_pow
    delta_exp = delta.unsqueeze(2)
    basis_exp = basis.view(1, 1, 8, 8, 1)
    deriv = n * (delta_exp * basis_exp).sum(dim=3)
    dx, dy = deriv[..., 0], deriv[..., 1]
    yaw = torch.atan2(dy, dx).unsqueeze(-1)
    return torch.cat([xy8, yaw], dim=-1)


def grpo_advantages_torch(
    reward_group: torch.Tensor,
    batched_sub: Dict[str, torch.Tensor],
    reward_gt: torch.Tensor,
    num_groups: int,
    ego_fut_mode: int,
    step_num: int = 10,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Truncated inter-anchor GRPO advantages (matches model_rl.forward_train_rl).

    reward_group: (B, G*N) without GT
    batched_sub: each (B, G*N+1) with GT last OR (B, G*N) without — if +1, strip last
    reward_gt: (B, 1)
    returns advantages (B, G*N, step_num), scalar mean reward for logging
    """
    bs = reward_group.shape[0]
    device = reward_group.device
    rg = reward_group.view(bs, num_groups, ego_fut_mode)
    # Intra-anchor: relative across groups (dim=1). If G==1, std is NaN — fall back to
    # relative across modes within the single group (dim=2).
    if num_groups >= 2:
        mean = rg.mean(dim=1, keepdim=True)
        std = torch.nan_to_num(rg.std(dim=1, keepdim=True), nan=0.0)
        advantages = (rg - mean) / (std + 1e-4)
    else:
        mean = rg.mean(dim=2, keepdim=True)
        std = torch.nan_to_num(rg.std(dim=2, keepdim=True), nan=0.0)
        advantages = (rg - mean) / (std + 1e-4)
    mask_positive = rg > (reward_gt.view(bs, 1, 1) - 1e-6)
    advantages = advantages.clamp(min=0) * mask_positive.float()
    advantages = torch.nan_to_num(advantages, nan=0.0)

    for k, v_full in batched_sub.items():
        if v_full.shape[1] == num_groups * ego_fut_mode + 1:
            v = v_full[:, :-1]
        else:
            v = v_full
        v = v.view(bs, num_groups, ego_fut_mode)
        if k in ("no_collision", "drivable_area"):
            zero_mask = v != 1
            advantages = torch.where(zero_mask, torch.full_like(advantages, -1.0), advantages)

    pos_cnt = mask_positive.sum(dim=(1, 2), keepdim=True)
    pos_sum = (rg * mask_positive.float()).sum(dim=(1, 2), keepdim=True)
    mean_pos = pos_sum / pos_cnt.clamp(min=1)
    mean_all = rg.mean(dim=(1, 2), keepdim=True)
    batch_reward = torch.where(pos_cnt > 0, mean_pos, mean_all)
    reward = batch_reward.squeeze(-1).mean()

    advantages = advantages.view(bs, num_groups * ego_fut_mode)
    advantages = advantages.detach().unsqueeze(-1).repeat(1, 1, step_num)
    discount = torch.tensor(
        [0.8 ** (step_num - i - 1) for i in range(step_num)],
        device=device,
    )
    advantages = advantages * discount
    return advantages, reward
