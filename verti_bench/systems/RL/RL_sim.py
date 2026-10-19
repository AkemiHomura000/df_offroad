import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr 

import os
import sys
import glob
import multiprocessing
import random
import numpy as np
import logging
import yaml
import argparse
from PIL import Image
import shutil

from verti_bench.rl.off_road_VertiBench import off_road_art
from stable_baselines3 import PPO

class RLSim:
    def __init__(self, config):
        if config['use_gui'] and not config['render']:
            raise ValueError("If use_gui is True, render must also be True. GUI requires rendering.")
            
        # Store configuration parameters
        self.config = config
        self.world_id = config['world_id']
        if not (1 <= self.world_id <= 100):
            raise ValueError(f"World ID must be between 1 and 100, got {self.world_id}")
        self.scale_factor = config['scale_factor']
        self.render = config['render']
        self.use_gui = config['use_gui']
        self.vehicle_type = config['vehicle']
        self.system_type = config['system']
        self.max_time = config['max_time']
        self.speed = config['speed']

        supported_vehicles = ['hmmwv']
        if self.vehicle_type.lower() not in supported_vehicles:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}, "
                             f"only support {supported_vehicles[0]} for RL-based systems!")
        
        # RL specific attributes
        self.env = None
        self.model = None
        self.obs = None
        self.total_steps = 0
        
        # Clean up tmp terrain directory
        terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../../envs/data/BenchMaps/sampled_maps/Configs/tmp")
        if os.path.exists(terrain_dir):
            shutil.rmtree(terrain_dir)
        
    def initialize(self):
        """Initialize the RL sim"""
        self.env = off_road_art(world_id=self.world_id, scale_factor=self.scale_factor)
        self.env.m_max_time = self.max_time
        self.env.max_speed = self.speed
        
        # Load the pre-trained RL model
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "ppo_RL_iter34_level45.zip")
        try:
            self.model = PPO.load(model_path, self.env)
            print(f"Loaded RL model from {model_path}")
        except Exception as e:
            print(f"Error loading RL model: {e}")
            raise
        
        # Reset the environment
        self.obs, self.info = self.env.reset()
        
        # Initialize visualization if rendering is enabled
        if self.render:
            self.env.render('follow')
        
        # Initialize tracking variables
        self.total_steps = 0

    def run(self):
        """Run the simulation"""
        if self.env is None or self.model is None:
            raise ValueError("Simulation not initialized. Call initialize() first.")
        
        self.total_steps = self.env.m_max_time / self.env.m_step_size
        
        for step in range(int(self.total_steps)):
            action, _states = self.model.predict(self.obs, deterministic=True)
            print(f"Step {step + 1}")
            print("Action: ", action)
            self.obs, reward, self.terminated, self.truncated, info = self.env.step(action)
            print("obs=", self.obs, "reward=", reward, "done=", (self.terminated or self.truncated))
            if self.render:
                self.env.render('follow')
            if self.terminated or self.truncated:
                break
            
        return info.get('time_to_goal'), info.get('success', False), info.get('roll_angles', []), info.get('pitch_angles', [])
    