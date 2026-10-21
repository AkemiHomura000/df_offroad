"""Optional adapter: swap PDMS scoring for VertiPDM inside an RL TrajectoryHead.

Official `diffusiondrivev2_model_rl.TrajectoryHead.get_pdm_score_para` cannot run in
the lean ddv2 env (nuplan/PDMS missing). Prefer `train_verti_grpo.py`, which calls
`get_verti_score_para` directly.

If you have a full NAVSIM stack and only want to replace the scorer backend:

  from patch_verti_pdm_backend import install_verti_pdm_backend
  install_verti_pdm_backend(trajectory_head)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "bridge"))
sys.path.insert(0, str(ROOT / "scripts" / "train"))

from verti_grpo_utils import WorldMapBank, get_verti_score_para  # noqa: E402


def install_verti_pdm_backend(trajectory_head: Any, bank: Optional[WorldMapBank] = None) -> None:
    """Monkey-patch get_pdm_score_para → get_verti_score_para (GT appended last)."""
    bank = bank or WorldMapBank()

    def _get_verti_score_para(self, trajectory, metric_cache_path):
        # metric_cache_path here is expected to be a list of VertiMetricCache / dicts
        return get_verti_score_para(trajectory, metric_cache_path, bank=bank, first_is_gt=False)

    trajectory_head.get_pdm_score_para = _get_verti_score_para.__get__(trajectory_head, type(trajectory_head))
    print("[patch] TrajectoryHead.get_pdm_score_para → VertiPDM get_verti_score_para")
