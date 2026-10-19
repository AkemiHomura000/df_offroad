#!/usr/bin/env python3
"""chrono9: short Verti-Bench run → export one DDV2-shaped observation."""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = "/home/rm/Desktop/df_offroad"
sys.path.insert(0, ROOT)

from verti_bench.envs.utils.utils import SetChronoDataDirectories
from verti_bench.systems.PID.PID_sim import PIDSim


def main():
    out = os.environ.get("BRIDGE_OBS", f"{ROOT}/log/bridge/obs.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    SetChronoDataDirectories()

    config = {
        "vehicle": "hmmwv",
        "speed": 4.0,
        "system": "pid",
        "world_id": 1,
        "max_time": 2.0,
        "scale_factor": 1.0,
        "render": False,
        "use_gui": False,
        "ddv2_sensors": True,
        "ddv2_dump_dir": None,
    }
    sim = PIDSim(config)
    sim.initialize()

    obs = None
    t0 = sim.system.GetChTime()
    while sim.system.GetChTime() - t0 < 1.5:
        for terrain in sim.terrains:
            terrain.Synchronize(sim.system.GetChTime())
            sim.vehicle_manager.synchronize(
                sim.system.GetChTime(), sim.driver_inputs, terrain
            )
            terrain.Advance(sim.step_size)
        sim.driver.Advance(sim.step_size)
        sim.vehicle_manager.advance(sim.step_size)
        sim.system.DoStepDynamics(sim.step_size)
        sim.ddv2_suite.update()
        cand = sim.ddv2_suite.get_observation(goal=sim.vehicle_manager.goal)
        if cand is not None:
            obs = cand

    if obs is None:
        raise RuntimeError("DDV2 sensors never produced data")

    np.savez_compressed(
        out,
        camera_feature=obs["camera_feature"],
        lidar_feature=obs["lidar_feature"],
        status_feature=obs["status_feature"],
    )
    print(
        f"[chrono] wrote {out}\n"
        f"  camera {obs['camera_feature'].shape} "
        f"lidar {obs['lidar_feature'].shape} "
        f"status {obs['status_feature'].shape}"
    )


if __name__ == "__main__":
    main()
