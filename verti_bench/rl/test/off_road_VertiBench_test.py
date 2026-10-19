import gymnasium as gym
from verti_bench.rl.off_road_VertiBench import off_road_art
from gymnasium.utils.env_checker import check_env
import shutil
import os

render = True
if __name__ == '__main__':
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
    if os.path.exists(terrain_dir):
        shutil.rmtree(terrain_dir)
        
    env = off_road_art(world_id=1, scale_factor=1.0)    
    obs, _ = env.reset()
    if render:
        env.render('follow')

    n_steps = 12000
    for step in range(n_steps):
        print(f"Step {step + 1}")
        '''
        Steering: -1 is right, 1 is left
        '''
        obs, reward, terminated, truncated, info = env.step([0.0, 1.0])
        print(obs, reward)
        print("Terminated=", terminated, "Truncated=", truncated)
        done = terminated or truncated
        if render:
            env.render('follow')
        if done:
            print("reward=", reward)
            break
