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

from verti_bench.envs.terrain import TerrainManager
from verti_bench.systems.TAL.TAL import TALPlanner
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.vehicles.FEDA import FEDAManager
from verti_bench.vehicles.Gator import GatorManager
from verti_bench.vehicles.MAN5t import MAN5tManager
from verti_bench.vehicles.MAN7t import MAN7tManager
from verti_bench.vehicles.MAN10t import MAN10tManager
from verti_bench.vehicles.M113 import M113Manager
from verti_bench.vehicles.ART import ARTManager
from verti_bench.vehicles.VW import VWManager

class TALSim:
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
        self.vehicle_type_lower = self.vehicle_type.lower()
        
        # Initialize system
        self.system = chrono.ChSystemNSC()
        self.system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
        self.system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
        
        # Set thread counts based on available CPUs
        num_procs = multiprocessing.cpu_count()
        num_threads_chrono = min(8, num_procs)
        num_threads_collision = min(8, num_procs)
        num_threads_eigen = 1
        self.system.SetNumThreads(num_threads_chrono, num_threads_collision, num_threads_eigen)
        
        # Simulation parameters
        self.step_size = self._step_size()
        self.vis_freq = 100.0
        self.vis_dur = 1.0 / self.vis_freq
        self.last_vis_time = 0.0
        self.mppi_freq = 20.0  # Hz
        self.mppi_dur = 1.0 / self.mppi_freq
        self.last_mppi_time = 0.0
        self.last_replan_time = 0.0
        self.replan_interval = 0.1
        self.vis = None
        self.driver = None
        
        # Stuck tracking
        self.stuck_counter = 0
        self.stuck_distance = 0.01
        self.stuck_time = self._stuck_time()
        self.last_position = None
        
        # Load terrain configs
        self.terrain_manager = TerrainManager(self.world_id, self.scale_factor)
            
        # Create planner
        self._initialize_system()
        
        # Create vehicle manager
        self._initialize_vehicle()
    
    def _step_size(self):
        """Get vehicle-specific step size based on vehicle type"""
        step_sizes = {
            'hmmwv': 5e-3,
            'gator': 2e-3,
            'feda': 1e-3,
            'man5t': 1e-3,
            'man7t': 1e-3,
            'man10t': 1e-3,
            'm113': 8e-4,
            'art': 1e-3,
            'vw': 3e-4,
            'default': 1e-3       
        }
        
        return step_sizes[self.vehicle_type_lower]
    
    def _stuck_time(self):
        """
        Get stuck time
        """
        stuck_time = {
            'hmmwv': 10,
            'gator': 40,
            'feda': 40,
            'man5t': 50,
            'man7t': 50,
            'man10t': 60,
            'm113': 60,
            'art': 60,
            'vw': 60,
            'default': 10       
        }
        
        return stuck_time[self.vehicle_type_lower]
        
    def _initialize_system(self):
        """Initialize the control system based on system_type"""
        if self.system_type.lower() == 'tal':
            self.planner = TALPlanner(self.terrain_manager)
        else:
            raise ValueError(f"Unsupported system type: {self.system_type}.")
    
    def _initialize_vehicle(self):
        """Initialize the vehicle manager based on vehicle_type"""
        if self.vehicle_type.lower() == 'hmmwv':
            self.vehicle_manager = HMMWVManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'gator':
            self.vehicle_manager = GatorManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'feda':
            self.vehicle_manager = FEDAManager(self.system, self.step_size)
        elif self.vehicle_type.lower() == 'man5t':
            self.vehicle_manager = MAN5tManager(self.system, self.step_size)    
        elif self.vehicle_type.lower() == 'man7t':
            self.vehicle_manager = MAN7tManager(self.system, self.step_size)    
        elif self.vehicle_type.lower() == 'man10t':
            self.vehicle_manager = MAN10tManager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'm113':
            self.vehicle_manager = M113Manager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'art':
            self.vehicle_manager = ARTManager(self.system, self.step_size)  
        elif self.vehicle_type.lower() == 'vw':
            self.vehicle_manager = VWManager(self.system, self.step_size)
        else:
            raise ValueError(f"Unsupported vehicle type: {self.vehicle_type}. ")
        
    def _setup_visualization(self):
        """Set up visualization system"""
        self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        if self.vehicle_type_lower in ['m113']:
            self.vis =veh.ChTrackedVehicleVisualSystemIrrlicht()
        self.vis.SetWindowTitle('vws in the wild')
        self.vis.SetWindowSize(3840, 2160)
        
        if self.vehicle_type_lower in ['man5t', 'man7t', 'man10t']:
            trackPoint = chrono.ChVector3d(-5.5, 0.0, 2.6)
        elif self.vehicle_type_lower in ['hmmwv', 'gator', 'feda', 'm113']:
            trackPoint = chrono.ChVector3d(-1.0, 0.0, 1.75)
        else:
            trackPoint = chrono.ChVector3d(2.0, 0.0, 0.0)
        self.vis.SetChaseCamera(trackPoint, 6.0, 0.5)
        self.vis.Initialize()
        self.vis.AddLightDirectional()
        self.vis.AddSkyBox()
        self.vis.AttachVehicle(self.vehicle_manager.vehicle.GetVehicle())
        self.vis.EnableStats(True)
        
    def _setup_driver(self):
        """Set up driver (interactive or autonomous)"""
        if self.use_gui:  
            self.driver = veh.ChInteractiveDriverIRR(self.vis)
            self.driver.SetSteeringDelta(0.1)
            self.driver.SetThrottleDelta(0.02)
            self.driver.SetBrakingDelta(0.06)
            self.driver.Initialize()
        else:
            self.driver = veh.ChDriver(self.vehicle_manager.vehicle.GetVehicle())
            
        self.driver_inputs = self.driver.GetInputs()
            
    def initialize(self, start_pos=None, goal_pos=None):
        """Initialize simulation"""
        # Use positions from config
        if start_pos is None or goal_pos is None:
            positions = self.terrain_manager.positions
            pos_id = random.randint(0, len(positions) - 1)
            selected_pair = positions[pos_id]
            start_pos = [i * self.scale_factor for i in selected_pair['start']]
            goal_pos = [i * self.scale_factor for i in selected_pair['goal']]
        
        # Initialize terrain
        self.terrains = self.terrain_manager.initialize_terrain(self.system)
        
        # Initialize vehicle
        self.vehicle_manager.initialize_vehicle(start_pos, goal_pos, self.terrain_manager)
        
        # Set up moving patches if needed
        if self.terrain_manager.terrain_type == 'deformable' or self.terrain_manager.terrain_type == 'mixed':
            if self.vehicle_type.lower() in ['m113']:
                deform_terrains = [t for t in self.terrains if isinstance(t, veh.SCMTerrain)]
                self.vehicle_manager.setup_moving_patches(deform_terrains, True)
            else:
                deform_terrains = [t for t in self.terrains if isinstance(t, veh.SCMTerrain)]
                self.vehicle_manager.setup_moving_patches(deform_terrains, False)
        
        # Create obstacle map and plan path
        obs_path = self.terrain_manager.obs_path
        obstacle_array = np.array(Image.open(obs_path))
        self.planner.set_obstacle_map(obstacle_array)
        self.chrono_path = self.planner.astar_path(obs_path, start_pos, goal_pos)
        self.local_goal_idx = 0
        
        # Set up visualization
        if self.render:
            self._setup_visualization()
            
        # Set up driver
        self._setup_driver()
            
    def run(self):
        """Run the simulation"""
        # Check initialization
        if not hasattr(self, 'terrains') or not self.terrains:
            raise ValueError("Simulation not initialized. Call initialize() first.")
            
        # Initialize timing
        start_time = self.system.GetChTime()
        roll_angles = []
        pitch_angles = []
        
        # Main simulation loop
        while True:
            time = self.system.GetChTime()
            
            # Handle visualization if enabled
            if self.render:
                if not self.vis.Run():
                    break
                    
                if self.last_vis_time == 0 or (time - self.last_vis_time) > self.vis_dur:
                    self.vis.BeginScene()
                    self.vis.Render()
                    self.vis.EndScene()
                    self.last_vis_time = time
            
            # Get vehicle position and orientation
            vehicle_pos = self.vehicle_manager.get_position()
            vector_to_goal = self.vehicle_manager.goal - vehicle_pos
            
            # Update controls
            if self.use_gui:
                self.driver_inputs = self.driver.GetInputs()
            else:
                # Get vehicle orientation
                euler_angles = self.vehicle_manager.get_rotation()
                roll = euler_angles.x
                pitch = euler_angles.y
                vehicle_heading = euler_angles.z
                roll_angles.append(np.degrees(abs(roll)))
                pitch_angles.append(np.degrees(abs(pitch)))
                
                # Check if replanning is needed
                if time - self.last_replan_time >= self.replan_interval:
                    print("Replanning path...")
                    obs_path = self.terrain_manager.obs_path
                    new_path = self.planner.astar_replan(
                        obs_path,
                        (vehicle_pos.x, vehicle_pos.y),
                        (self.vehicle_manager.goal.x, self.vehicle_manager.goal.y)
                    )
                    
                    if new_path is not None:
                        self.chrono_path = new_path
                        self.local_goal_idx = 0
                        print("Path replanned successfully")
                    else:
                        print("Failed to replan path!")
                    
                    self.last_replan_time = time
                
                if self.last_mppi_time == 0 or (time - self.last_mppi_time) >= self.mppi_dur:
                    # Find local goal along the path
                    self.local_goal_idx, local_goal = self.planner.find_local_goal(
                        (vehicle_pos.x, vehicle_pos.y), 
                        vehicle_heading,
                        self.chrono_path, 
                        self.local_goal_idx
                    )

                    # Update goal and odometry in MPPI
                    self.planner.goal_cb(local_goal)
                    self.planner.odom_cb(self.vehicle_manager, self.system)
                    
                    # Update elevation map
                    obstacle_array = np.array(Image.open(self.terrain_manager.obs_path))
                    self.planner.gridMap_callback(self.vehicle_manager.vehicle, vehicle_pos, obstacle_array)
                    
                    # Run MPPI update
                    self.planner.mppi_cb(self.planner.curr_pose, self.planner.pose_dot)
                    
                    # Get control commands
                    speed, steer = self.planner.send_controls()
                    
                    # Scale speed appropriately for the vehicle
                    speed = speed * 10.0
                    
                    self.last_mppi_time = time
                    
                # Apply steering with rate limiting
                self.driver_inputs.m_steering = np.clip(steer, -1, 1)
                
                # Compute throttle and braking
                throttle, braking = self.planner.compute_throttle(
                    self.speed, 
                    time, 
                    self.step_size,
                    self.vehicle_manager.vehicle.GetVehicle().GetRefFrame()
                )
                
                self.driver_inputs.m_throttle = throttle
                self.driver_inputs.m_braking = braking
            
            # Check if vehicle is stuck or reached goal
            current_position = (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z)
            
            if self.last_position:
                position_change = np.sqrt(
                    (current_position[0] - self.last_position[0])**2 +
                    (current_position[1] - self.last_position[1])**2 +
                    (current_position[2] - self.last_position[2])**2
                )
                
                if position_change < self.stuck_distance:
                    self.stuck_counter += self.step_size
                else:
                    self.stuck_counter = 0
                    
                if self.stuck_counter >= self.stuck_time:
                    print('--------------------------------------------------------------')
                    print('Vehicle stuck!')
                    print(f'Stuck time: {self.stuck_counter:.2f} seconds')
                    print(f'Position change: {position_change:.3f} m')
                    print(f'Initial position: {self.vehicle_manager.init_loc}')
                    print(f'Current position: {vehicle_pos}')
                    print(f'Goal position: {self.vehicle_manager.goal}')
                    print(f'Distance to goal: {vector_to_goal.Length():.2f} m')
                    print('--------------------------------------------------------------')
                    
                    if self.render:
                        self.vis.Quit()
                        
                    avg_roll = np.mean(roll_angles) if roll_angles else 0
                    avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                    return time - start_time, False, avg_roll, avg_pitch
            
            self.last_position = current_position
            
            # Check if goal reached
            if vector_to_goal.Length() < 8 * self.scale_factor:
                print('--------------------------------------------------------------')
                print('Goal Reached')
                print(f'Initial position: {self.vehicle_manager.init_loc}')
                print(f'Goal position: {self.vehicle_manager.goal}')
                print('--------------------------------------------------------------')
                
                if self.render:
                    self.vis.Quit()
                    
                avg_roll = np.mean(roll_angles) if roll_angles else 0 
                avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                return time - start_time, True, avg_roll, avg_pitch
            
            # Check if time limit exceeded
            if time > self.max_time:
                print('--------------------------------------------------------------')
                print('Time out')
                print('Initial position: ', self.vehicle_manager.init_loc)
                dist = vector_to_goal.Length()
                print('Final position of vw: ', self.vehicle_manager.chassis_body.GetPos())
                print('Goal position: ', self.vehicle_manager.goal)
                print('Distance to goal: ', dist)
                print('--------------------------------------------------------------')
                
                if self.render:
                    self.vis.Quit()
                    
                avg_roll = np.mean(roll_angles) if roll_angles else 0 
                avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                return time - start_time, False, avg_roll, avg_pitch
            
            # Synchronize terrains and vehicle
            for terrain in self.terrains:
                terrain.Synchronize(time)
                
                if self.vehicle_type_lower in ['m113']:
                    self.vehicle_manager.synchronize(time, self.driver_inputs)
                else:
                    self.vehicle_manager.synchronize(time, self.driver_inputs, terrain)
                
                terrain.Advance(self.step_size)
            
            # Advance simulation components
            self.driver.Advance(self.step_size)
            self.vehicle_manager.advance(self.step_size)
            
            if self.render:
                self.vis.Synchronize(time, self.driver_inputs)
                self.vis.Advance(self.step_size)
            
            # Step the system
            self.system.DoStepDynamics(self.step_size)
        
        return None, False, 0, 0  # Return default values if loop exits unexpectedly
    