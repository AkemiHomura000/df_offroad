import gymnasium as gym
from stable_baselines3 import PPO
from verti_bench.rl.off_road_VertiBench import off_road_art
from gymnasium.utils.env_checker import check_env

import numpy as np
import os
import shutil

render = True
if __name__ == '__main__':
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
    if os.path.exists(terrain_dir):
        shutil.rmtree(terrain_dir)
        
    env = off_road_art(world_id=91, scale_factor=1.0)
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ppo_MCL_stage3_iter19.zip")
    loaded_model = PPO.load(model_path, env)

    #Render and test model
    totalSteps = 12000
    obs, _ = env.reset()
    if render:
        env.render('follow')
    for step in range(totalSteps):
        action, _states = loaded_model.predict(obs, deterministic=True)
        print(f"Step {step + 1}")
        print("Action: ", action)
        obs, reward, terminated, truncated, info = env.step(action)
        print("obs=", obs, "reward=", reward, "done=", (terminated or truncated))
        if render:
            env.render('follow')
        if (terminated or truncated):
            break
