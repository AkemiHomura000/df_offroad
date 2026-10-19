import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import math
import cv2

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens
from verti_bench.envs.utils.asset_utils import *

from PIL import Image, ImageDraw
import os
import shutil
import matplotlib.pyplot as plt
import copy
import yaml
import logging
import heapq
from scipy.ndimage import binary_dilation
from scipy.special import comb
import glob
import csv
import json
import multiprocessing
import datetime
import pickle
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from collections import defaultdict
from verti_bench.systems.Manual.ManualInput import ManualInput
import threading
import argparse
import sys

class SCMParameters:
    def __init__(self):
        self.Bekker_Kphi = 0    # Kphi, frictional modulus in Bekker model
        self.Bekker_Kc = 0      # Kc, cohesive modulus in Bekker model
        self.Bekker_n = 0       # n, exponent of sinkage in Bekker model (usually 0.6...1.8)
        self.Mohr_cohesion = 0  # Cohesion in, Pa, for shear failure
        self.Mohr_friction = 0  # Friction angle (in degrees!), for shear failure
        self.Janosi_shear = 0   # J , shear parameter, in meters, in Janosi-Hanamoto formula (usually few mm or cm)
        self.elastic_K = 0      # elastic stiffness K (must be > Kphi very high values gives the original SCM model)
        self.damping_R = 0      # vertical damping R, per unit area (vertical speed proportional, it is zero in original SCM model)

    # Set the parameters of the terrain
    def SetParameters(self, terrain):
        terrain.SetSoilParameters(
            self.Bekker_Kphi,    # Bekker Kphi
            self.Bekker_Kc,      # Bekker Kc
            self.Bekker_n,       # Bekker n exponent
            self.Mohr_cohesion,  # Mohr cohesive limit (Pa)
            self.Mohr_friction,  # Mohr friction limit (degrees)
            self.Janosi_shear,   # Janosi shear coefficient (m)
            self.elastic_K,      # Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
            self.damping_R)      # Damping (Pa s/m), proportional to negative vertical speed (optional)

    # Soft default parameters
    def InitializeParametersAsSoft(self): # snow
        self.Bekker_Kphi = 0.2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 4e7
        self.damping_R = 3e4

    # Middle default parameters
    def InitializeParametersAsMid(self): # mud
        self.Bekker_Kphi = 2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 2e8
        self.damping_R = 3e4
    
    # Hard default parameters
    def InitializeParametersAsHard(self): # sand
        self.Bekker_Kphi = 5301e3
        self.Bekker_Kc = 102e3
        self.Bekker_n = 0.793
        self.Mohr_cohesion = 1.3e3
        self.Mohr_friction = 31.1
        self.Janosi_shear = 1.2e-2
        self.elastic_K = 4e8
        self.damping_R = 3e4
        
def deformable_params(terrain_type):
    """
    Initialize SCM parameters based on terrain type.
    Returns initialized SCMParameters object.
    """
    terrain_params = SCMParameters()
    
    if terrain_type == 'snow':
        terrain_params.InitializeParametersAsSoft()
    elif terrain_type == 'mud':
        terrain_params.InitializeParametersAsMid()
    elif terrain_type == 'sand':
        terrain_params.InitializeParametersAsHard()
    else:
        raise ValueError(f"Unknown deformable terrain type: {terrain_type}")
        
    return terrain_params

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

class SineSteer:
    def __init__(self, freq):
        self.freq = freq

    def advance(self, t): # sin(wt)
        return np.sin(self.freq * t) # [-1, 1]

class SineSpeed:
    def __init__(self, freq):
        self.freq = freq

    def advance(self, t):
        return 2.0 * (np.sin(self.freq * t) + 1.0) # [0, 4]

class AStarPlanner:
    def __init__(self, grid_map, start, goal, resolution=1.0):
        self.grid_map = grid_map
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.resolution = resolution
        self.open_list = []
        self.closed_list = set()
        self.parent = {}
        self.g_costs = {}

    def heuristic(self, node):
        return ((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)**0.5

    def get_neighbors(self, node):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (-1, 1), (-1, -1), (1, -1)]
        neighbors = []
        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if (0 <= neighbor[0] < self.grid_map.shape[0] and
                0 <= neighbor[1] < self.grid_map.shape[1] and
                self.grid_map[neighbor] == 0):
                neighbors.append(neighbor)
        return neighbors

    def plan(self):
        start_node = self.start
        goal_node = self.goal
        heapq.heappush(self.open_list, (0, start_node))
        self.g_costs[start_node] = 0

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            if current in self.closed_list:
                continue

            self.closed_list.add(current)

            if current == goal_node:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_cost = self.g_costs[current] + self.resolution
                if (neighbor not in self.g_costs or
                        tentative_g_cost < self.g_costs[neighbor]):
                    self.g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (f_cost, neighbor))
                    self.parent[neighbor] = current

        return None

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(self.start)
        return path[::-1]

# Use Bezier curve to smooth the path
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(points, num_samples=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num_samples)
    curve = np.zeros((num_samples, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), points[i])
    return curve

def smooth_path_bezier(path, num_samples=100):
    if path is None or len(path) < 3:
        return path
    
    points = np.array(path)
    curve = bezier_curve(points, num_samples)
    return curve

def chrono_to_grid(pos, inflated_map, m_terrain_length, m_terrain_width):
    """Convert Chrono coordinates to grid coordinates"""
    grid_height, grid_width = inflated_map.shape
    s_norm_x = grid_width / (2 * m_terrain_length)
    s_norm_y = grid_height / (2 * m_terrain_width)
    T = np.array([
        [s_norm_x, 0, 0],
        [0, s_norm_y, 0],
        [0, 0, 1]
    ])
    vehicle_x = pos[0]  # PyChrono X (Forward)
    vehicle_y = -pos[1]  # PyChrono Y (Left)
    pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])
    res = np.dot(T, pos_chrono)
    return (min(max(int(res[1]), 0), grid_height - 1), min(max(int(res[0]), 0), grid_width - 1))
        
def astar_path(obs_path, start_pos, goal_pos):
    """
    A* path on the obstacle map
    """
    # Load obstacle map
    obs_image = Image.open(obs_path)
    obs_map = np.array(obs_image.convert('L'))
    grid_map = np.where(obs_map == 255, 1, 0)
    structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1))
    inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
    
    # Convert start and goal positions
    start_grid = chrono_to_grid(start_pos, inflated_map, m_terrain_length, m_terrain_width)
    goal_grid = chrono_to_grid(goal_pos, inflated_map, m_terrain_length, m_terrain_width)
    
    # Plan path
    planner = AStarPlanner(inflated_map, start_grid, goal_grid)
    path = planner.plan()
    path = smooth_path_bezier(path)
    
    if path is None:
        print("No valid path found!")
        return None
    
    # # Plotting the path
    # plt.style.use('default')
    # plt.rcParams['font.family'] = 'DejaVu Serif'
    # plt.rcParams['font.size'] = 20
    # plt.rcParams['axes.titlesize'] = 20
    # plt.rcParams['axes.labelsize'] = 20
    # plt.rcParams['xtick.labelsize'] = 14
    # plt.rcParams['ytick.labelsize'] = 14
    
    # # Create figure
    # fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    
    # # Set background colors
    # ax.set_facecolor('#F0F0F0')
    # fig.patch.set_facecolor('white')
    
    # # Remove tick marks while keeping labels
    # ax.tick_params(axis='both', length=0)

    # # Plot the obstacle map
    # plt.imshow(grid_map, cmap='binary', alpha=1.0, vmin=0, vmax=1, 
    #            extent=[0, 129 * SCALE_FACTOR, 129 * SCALE_FACTOR, 0])
    
    # # Plot the smooth path with professional styling
    # path_y = [p[0] * SCALE_FACTOR for p in path]
    # path_x = [p[1] * SCALE_FACTOR for p in path]
    # plt.plot(path_x, path_y, color='#D62728', linewidth=2, zorder=3)

    # # Plot start and goal with consistent styling
    # plt.scatter(start_grid[1] * SCALE_FACTOR, start_grid[0] * SCALE_FACTOR, 
    #             color='green', s=150, zorder=4, edgecolor='green', linewidth=1.5)
    # plt.scatter(goal_grid[1] * SCALE_FACTOR, goal_grid[0] * SCALE_FACTOR, 
    #             color='red', s=200, marker='*', zorder=4, edgecolor='darkred', linewidth=1.5)

    # # Set axis limits
    # plt.xlim(0, 129 * SCALE_FACTOR)
    # plt.ylim(129 * SCALE_FACTOR, 0)
    
    # # Customize axes labels
    # ax.set_xlabel('X Position (m)', weight='bold', labelpad=10)
    # ax.set_ylabel('Y Position (m)', weight='bold', labelpad=10)
    
    # # Remove spines and add grid
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_color('white')
    # ax.spines['bottom'].set_color('white')
    
    # # Set aspect ratio and layout
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
    
    # Convert to Chrono coordinates
    bitmap_points = [(point[1], point[0]) for point in path]
    chrono_path = transform_to_chrono(bitmap_points)
    
    return chrono_path

def astar_replan(obs_path, current_pos, goal_pos):
    """
    A* replan from current position to goal
    """
    # Load obstacle map
    obs_image = Image.open(obs_path)
    obs_map = np.array(obs_image.convert('L'))
    grid_map = np.where(obs_map == 255, 1, 0)
    structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1))
    inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
    
    current_grid = chrono_to_grid(current_pos, inflated_map, m_terrain_length, m_terrain_width)
    goal_grid = chrono_to_grid(goal_pos, inflated_map, m_terrain_length, m_terrain_width)
    
    # Replan path
    planner = AStarPlanner(inflated_map, current_grid, goal_grid)
    path = planner.plan()
    
    if path is None:
        print("No valid path found in replanning!")
        return None
    
    if len(path) >= 3:
        path = smooth_path_bezier(path)
        
    bitmap_points = [(point[1], point[0]) for point in path]
    chrono_path = transform_to_chrono(bitmap_points)
    
    return chrono_path

def transform_to_bmp(chrono_positions):
    bmp_dim_y, bmp_dim_x = terrain_array.shape
    
    # Normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)
    
    # Transformation matrix
    T = np.array([
        [s_norm_x, 0, 0],
        [0, s_norm_y, 0],
        [0, 0, 1]
    ])

    bmp_positions = []
    for pos in chrono_positions:
        # Adjust PyChrono coordinates
        vehicle_x = pos[0]  
        vehicle_y = -pos[1] 
        pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])

        # Transform to BMP coordinates
        pos_bmp = np.dot(T, pos_chrono)
        bmp_positions.append((pos_bmp[0], pos_bmp[1]))

    return bmp_positions

def transform_to_chrono(bmp_positions):
    bmp_dim_y, bmp_dim_x = terrain_array.shape
        
    # Inverse normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)

    # Inverse transformation matrix
    T_inv = np.array([
        [1 / s_norm_x, 0, 0],
        [0, 1 / s_norm_y, 0],
        [0, 0, 1]
    ])

    chrono_positions = []
    for pos in bmp_positions:
        pos_bmp = np.array([pos[0], pos[1], 1])
        pos_chrono = np.dot(T_inv, pos_bmp)

        # Adjust to PyChrono coordinate system
        x = pos_chrono[0] - m_terrain_length
        y = -(pos_chrono[1] - m_terrain_width)
        chrono_positions.append((x, y))

    return chrono_positions

def transform_to_high_res(chrono_positions, height_array=None):
    """Transform PyChrono coordinates to high-res bitmap coordinates"""
    if height_array is None:
        height_array = high_res_data
        
    bmp_dim_y, bmp_dim_x = height_array.shape
    
    # Normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)
    
    # Transformation matrix
    T = np.array([
        [s_norm_x, 0, 0],
        [0, s_norm_y, 0],
        [0, 0, 1]
    ])
    
    bmp_positions = []
    for pos in chrono_positions:
        vehicle_x = pos[0]  
        vehicle_y = -pos[1] 
        pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])
        
        # Transform to BMP coordinates
        pos_bmp = np.dot(T, pos_chrono)
        bmp_positions.append((pos_bmp[0], pos_bmp[1]))
    
    return bmp_positions

def find_local_goal(vehicle_pos, vehicle_heading, chrono_path, local_goal_idx):
    """Find local goal point along the path based on look-ahead distance"""
    if not chrono_path:
        print(f"Warning: no valid chrono path!")
        current_x, current_y = vehicle_pos[0], vehicle_pos[1]
        return local_goal_idx, (current_x, current_y)

    if local_goal_idx >= len(chrono_path) - 1:
        final_idx = len(chrono_path) - 1
        return final_idx, chrono_path[final_idx]
    
    for idx in range(local_goal_idx, len(chrono_path)):
        path_point = chrono_path[idx]
        distance = ((vehicle_pos[0] - path_point[0])**2 + (vehicle_pos[1] - path_point[1])**2)**0.5
        
        if distance >= look_ahead_distance:
            dx = path_point[0] - vehicle_pos[0]
            dy = path_point[1] - vehicle_pos[1]
            angle_to_point = np.arctan2(dy, dx)
            angle_diff = (angle_to_point - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_diff) <= np.pi / 2:
                return idx, path_point
            
    final_idx = len(chrono_path) - 1
    return final_idx, chrono_path[final_idx]

def initialize_vw_pos(m_vehicle, start_pos, goal_pos, m_isFlat):
    """Initialize the vehicle position and orientation"""
    if m_isFlat:
        start_height = 0
    else:
        # Get height from terrain at start position
        pos_bmp = transform_to_high_res([start_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1))) 
        start_height = high_res_data[pos_bmp_y, pos_bmp_x]

    # Set position with correct height
    start_pos = (start_pos[0], start_pos[1], start_height * SCALE_FACTOR + start_pos[2])
    
    # Calculate orientation based on direction to goal
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    start_yaw = np.arctan2(dy, dx)
    
    # Create location and rotation objects
    init_loc = chrono.ChVector3d(*start_pos)
    init_rot = chrono.QuatFromAngleZ(start_yaw)
    
    # Set position
    m_vehicle.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
    
    return init_loc, init_rot, start_yaw

def set_goal(m_system, goal_pos, m_isFlat):
    """Create a goal marker at the target position"""
    if m_isFlat:
        goal_height = 0
    else:
        # Get height from terrain at goal position
        pos_bmp = transform_to_high_res([goal_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1)))
        goal_height = high_res_data[pos_bmp_y, pos_bmp_x]
    
    # Set goal position with correct height
    offset = 1.0 * SCALE_FACTOR
    goal_pos = (goal_pos[0], goal_pos[1], goal_height * SCALE_FACTOR + goal_pos[2] + offset)
    goal = chrono.ChVector3d(*goal_pos)

    # Create goal sphere with visualization
    goal_contact_material = chrono.ChContactMaterialNSC()
    goal_body = chrono.ChBodyEasySphere(0.5 * SCALE_FACTOR, 1000, True, False, goal_contact_material)
    goal_body.SetPos(goal)
    goal_body.SetFixed(True)
    
    # Apply red visualization material
    goal_mat = chrono.ChVisualMaterial()
    goal_mat.SetAmbientColor(chrono.ChColor(1, 0, 0)) 
    goal_mat.SetDiffuseColor(chrono.ChColor(1, 0, 0))
    goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
    
    # Add goal to system
    m_system.Add(goal_body)
    
    return goal

def get_cropped_elev(vehicle, vehicle_pos, region_size, num_front_regions):
    """Get terrain height maps around the vehicle"""
    bmp_dim_y, bmp_dim_x = high_res_data.shape  # height (rows), width (columns)
    pos_bmp = transform_to_high_res([vehicle_pos])[0]
    pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
    pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1)))
    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < high_res_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < high_res_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    # Shift the map to center the vehicle position
    shifted_map = np.roll(high_res_data, shift_y, axis=0)  # y shift affects rows (axis 0)
    shifted_map = np.roll(shifted_map, shift_x, axis=1)    # x shift affects columns (axis 1)

    # Rotate the map based on vehicle heading
    vehicle_heading_global = vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
    angle = np.degrees(vehicle_heading_global) % 360
    
    # Using tensor to accelerate the rotation process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_map = torch.tensor(shifted_map, device=device).unsqueeze(0).float()
    rotated_tensor = F.rotate(tensor_map, -angle)
    rotated_map = rotated_tensor.squeeze().cpu().numpy()
    rotated_map = np.fliplr(rotated_map)

    # Extract the part under the vehicle
    center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
    under_vehicle_start_y = center_y - region_size // 2
    under_vehicle_end_y = center_y + region_size // 2
    under_vehicle_start_x = center_x - region_size // 2
    under_vehicle_end_x = center_x + region_size // 2
    
    # Handle boundary conditions for under_vehicle
    under_vehicle_start_x = max(0, under_vehicle_start_x)
    under_vehicle_end_x = min(rotated_map.shape[1], under_vehicle_end_x)
    under_vehicle_start_y = max(0, under_vehicle_start_y)
    under_vehicle_end_y = min(rotated_map.shape[0], under_vehicle_end_y)
    under_vehicle = rotated_map[
        under_vehicle_start_y:under_vehicle_end_y,
        under_vehicle_start_x:under_vehicle_end_x
    ]
    under_vehicle = under_vehicle.T
    
    # Extract the part in front of the vehicle
    front_regions = []
    offset = num_front_regions // 2
    for i in range(-offset, offset+1):
        front_start_y = under_vehicle_start_y - region_size
        front_end_y = under_vehicle_start_y
        front_start_x = center_x - region_size // 2 + i * region_size
        front_end_x = front_start_x + region_size
        
        # Handle boundary conditions for front regions
        front_start_x = max(0, front_start_x)
        front_end_x = min(rotated_map.shape[1], front_end_x)
        front_start_y = max(0, front_start_y)
        front_end_y = min(rotated_map.shape[0], front_end_y)       
        
        front_region = rotated_map[
            front_start_y:front_end_y,
            front_start_x:front_end_x
        ]
        front_region = front_region.T
        front_regions.append(front_region)
        
    return under_vehicle, front_regions

def get_cropped_sem(vehicle, vehicle_pos, region_size):
    """Get terrain semantic maps around the vehicle"""
    bmp_dim_y, bmp_dim_x = high_sem_data.shape[:2]  # height (rows), width (columns)
    pos_bmp = transform_to_high_res([vehicle_pos])[0]
    pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
    pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1)))
    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < high_res_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < high_res_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    # Shift the map to center the vehicle position
    shifted_map = np.roll(high_sem_data, shift_y, axis=0)  # y shift affects rows (axis 0)
    shifted_map = np.roll(shifted_map, shift_x, axis=1)    # x shift affects columns (axis 1)

    # Rotate the map based on vehicle heading
    vehicle_heading_global = vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
    angle = np.degrees(vehicle_heading_global) % 360
    
    # Using tensor to accelerate the rotation process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_map = torch.tensor(shifted_map, device=device).permute(2, 0, 1).float()
    rotated_tensor = F.rotate(tensor_map, -angle)
    rotated_map = rotated_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    rotated_map = np.fliplr(rotated_map)

    # Extract the part under the vehicle
    center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
    under_vehicle_start_y = center_y - region_size // 2
    under_vehicle_end_y = center_y + region_size // 2
    under_vehicle_start_x = center_x - region_size // 2
    under_vehicle_end_x = center_x + region_size // 2
    
    # Handle boundary conditions for under_vehicle
    under_vehicle_start_x = max(0, under_vehicle_start_x)
    under_vehicle_end_x = min(rotated_map.shape[1], under_vehicle_end_x)
    under_vehicle_start_y = max(0, under_vehicle_start_y)
    under_vehicle_end_y = min(rotated_map.shape[0], under_vehicle_end_y)
    under_vehicle = rotated_map[
        under_vehicle_start_y:under_vehicle_end_y,
        under_vehicle_start_x:under_vehicle_end_x
    ]
    under_vehicle = np.transpose(under_vehicle, (1, 0, 2))  # (height, width, channels) -> (width, height, channels)
    return under_vehicle

def save_cropped_maps(elevation_map, semantic_map, timestr, timestep):
    """
    Save cropped elevation and semantic maps for current timestep
    """
    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                           "./data/BenchMaps/sampled_maps/Expert", str(world_id), f"{pos_id}_{timestr}")
    elev_dir = os.path.join(base_dir, "elev")
    sem_dir  = os.path.join(base_dir, "sem")
    os.makedirs(elev_dir, exist_ok=True)
    os.makedirs(sem_dir,  exist_ok=True)
    
    if elevation_map.max() > elevation_map.min():
        elevation_normalized = ((elevation_map - elevation_map.min()) * 255.0 / 
                               (elevation_map.max() - elevation_map.min())).astype(np.uint8)
    else:
        elevation_normalized = np.zeros_like(elevation_map, dtype=np.uint8)
        
    elev_image = Image.fromarray(elevation_normalized, mode='L')
    elev_filename = f"elev_{timestep:04d}.bmp"
    elev_path = os.path.join(elev_dir, elev_filename)
    elev_image.save(elev_path, format="BMP")
    
    sem_image = Image.fromarray(semantic_map, mode='RGB')
    sem_filename = f"sem_{timestep:04d}.png"
    sem_path = os.path.join(sem_dir, sem_filename)
    sem_image.save(sem_path, format="PNG")

def get_current_label(vehicle, vehicle_pos, region_size, high_res_terrain_labels):
    """Get terrain type labels beneath the vehicle"""
    bmp_dim_y, bmp_dim_x = high_res_terrain_labels.shape  # height (rows), width (columns)
    pos_bmp = transform_to_high_res([vehicle_pos], high_res_terrain_labels)[0]
    pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, bmp_dim_x - 1)))
    pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, bmp_dim_y - 1)))
    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    # Shift the map to center the vehicle position
    shifted_labels = np.roll(high_res_terrain_labels, shift_y, axis=0) 
    shifted_labels = np.roll(shifted_labels, shift_x, axis=1) 
    
    # Rotate the map based on vehicle heading
    vehicle_heading_global = vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
    angle = np.degrees(vehicle_heading_global) % 360
    
    # Using tensor to accelerate the rotation process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_labels = torch.tensor(shifted_labels, device=device).unsqueeze(0).float()
    rotated_tensor = F.rotate(tensor_labels, -angle)
    rotated_labels = rotated_tensor.squeeze().cpu().numpy().astype(np.int32)
    rotated_labels = np.fliplr(rotated_labels) 

    # Extract the part under the vehicle
    center_y, center_x = rotated_labels.shape[0] // 2, rotated_labels.shape[1] // 2
    start_y = center_y - region_size // 2
    end_y = center_y + region_size // 2
    start_x = center_x - region_size // 2
    end_x = center_x + region_size // 2
    
    # Handle boundary conditions
    start_y = max(0, start_y)
    end_y = min(rotated_labels.shape[0], end_y)
    start_x = max(0, start_x)
    end_x = min(rotated_labels.shape[1], end_x)
    
    cropped_labels = rotated_labels[start_y:end_y, start_x:end_x]
    cropped_labels = cropped_labels.T
    return cropped_labels

def find_regular_shape(patch_size, max_dim):
    """
    Generates a list of possible rectangular shapes (width, height) that can be formed.
    The shapes are sorted by area in descending order to prioritize larger continuous regions.
    """
    if patch_size > max_dim:
        return []
    
    shapes = []
    max_patches = (max_dim - 1) // (patch_size - 1) + 1
    
    for width_patches in range(1, max_patches + 1):
        for height_patches in range(1, max_patches + 1):
            # Convert patch counts to actual dimensions with overlap
            if width_patches == 1:
                width = patch_size
            else:
                width = (width_patches - 1) * (patch_size - 1) + patch_size
                
            if height_patches == 1:
                height = patch_size
            else:
                height = (height_patches - 1) * (patch_size - 1) + patch_size
            
            # Check if shape fits within maximum dimension
            if width <= max_dim and height <= max_dim:
                shape = (width, height)
                if shape not in shapes:
                    shapes.append(shape)
                    
    # Sort shapes by area in descending order                
    shapes.sort(key=lambda x: x[0] * x[1], reverse=True)
    return shapes

def best_shape_fit(shapes, patch_size, available_patches):
    """
    Find the largest rectangular shape that can fit entirely within the available patches.
    """
    if not available_patches:
        return None, set()
        
    max_i = max(i for i, _ in available_patches)
    max_j = max(j for _, j in available_patches)
    
    # Try each shape from largest to smallest
    for width, height in shapes:
        # Calculate how many patches fit in this shape accounting for overlap
        if width == patch_size:
            patches_width = 1
        else:
            patches_width = (width - patch_size) // (patch_size - 1) + 1
            
        if height == patch_size:
            patches_height = 1
        else:
            patches_height = (height - patch_size) // (patch_size - 1) + 1
        
        # Skip if shape is too big for available grid
        if patches_width > max_j + 1 or patches_height > max_i + 1:
            continue
        
        # Try each possible top-left starting position
        for i_start in range(max_i - patches_height + 2):
            for j_start in range(max_j - patches_width + 2):
                # Check if all patches in the rectangle are available
                current_patches = {(i_start + i, j_start + j) 
                                for i in range(patches_height) 
                                for j in range(patches_width)}
                if current_patches.issubset(available_patches):
                    return (width, height), current_patches
                    
    return None, set()

def terrain_patch_bmp(terrain_array, start_y, end_y, start_x, end_x, idx):
    """Create bitmap file for a terrain patch"""
    # Boundary check
    if (start_y < 0 or end_y > terrain_array.shape[0] or
        start_x < 0 or end_x > terrain_array.shape[1]):
        raise ValueError("Indices out of bounds for terrain array")
    
    # Extract the patch
    patch_array = terrain_array[start_y:end_y, start_x:end_x]

    # Normalize and convert to uint8
    if patch_array.dtype != np.uint8:
        patch_array = ((patch_array - patch_array.min()) * (255.0 / (patch_array.max() - patch_array.min()))).astype(np.uint8)
    # Convert to PIL Image
    patch_image = Image.fromarray(patch_array, mode='L')
    
    # Create file path
    patch_file = f"terrain_patch_{idx}.bmp"
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/tmp")
    
    # Clean up previous tmp directory
    if os.path.exists(terrain_dir):
        shutil.rmtree(terrain_dir)
    
    os.makedirs(terrain_dir, exist_ok=True)
    terrain_path = os.path.join(terrain_dir, patch_file)
    
    # Save the image for deformable terrain
    try:
        patch_image.save(terrain_path, format="BMP")
        logging.info(f"Saved terrain patch to {terrain_path}")
    except Exception as e:
        logging.error(f"Failed to save terrain patch: {e}")
        raise
    
    return terrain_path

def collect_traj(vehicle, driver_inputs, time, wait_time_before_log, last_log_time, 
                 log_dur, timestr, target_speed, target_steering):
    """
    Collect trajectory data for current timestep
    Returns state, action, and whether data was logged
    """
    # Check if enough time has passed since simulation start
    if time < wait_time_before_log:
        return None, None, False, last_log_time
    
    # Check if enough time has passed since last log
    if last_log_time is None or (time - last_log_time) >= log_dur:
        # Get vehicle state
        pos = vehicle.GetVehicle().GetPos()
        chassis_body = vehicle.GetVehicle().GetChassisBody()
        vel = chassis_body.GetPosDt()
        euler_angles = vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ()
        roll_rate = vehicle.GetVehicle().GetRollRate()
        pitch_rate = vehicle.GetVehicle().GetPitchRate() 
        yaw_rate = vehicle.GetVehicle().GetYawRate()
        
        # Get real vehicle control inputs
        real_speed = vehicle.GetVehicle().GetSpeed()
        
        left_steer_angle = vehicle.GetVehicle().GetSteeringAngle(0, 0) 
        right_steer_angle = vehicle.GetVehicle().GetSteeringAngle(0, 1)
        avg_steer_angle = (left_steer_angle + right_steer_angle) / 2.0
        real_steering = np.clip(avg_steer_angle / vehicle.GetVehicle().GetMaxSteeringAngle(), -1.0, 1.0)

        # Get gear
        gear = vehicle.GetVehicle().GetTransmission().GetCurrentGear()
        
        # State: (x, y, z, x_dot, y_dot, z_dot, roll, pitch, yaw, roll_dot, pitch_dot, yaw_dot)
        state = {
            'x': pos.x,
            'y': pos.y, 
            'z': pos.z,
            'x_dot': vel.x,
            'y_dot': vel.y,
            'z_dot': vel.z,
            'roll': euler_angles.x,
            'pitch': euler_angles.y,
            'yaw': euler_angles.z,
            'roll_dot': roll_rate,
            'pitch_dot': pitch_rate,
            'yaw_dot': yaw_rate
        }
        
        region_size = 128
        under_vehicle_elev, _ = get_cropped_elev(vehicle, (pos.x, pos.y, pos.z), region_size, 5)
        under_vehicle_sem = get_cropped_sem(vehicle, (pos.x, pos.y, pos.z), region_size)
        timestep = int((time - wait_time_before_log) * log_freq) if time >= wait_time_before_log else 0
        save_cropped_maps(under_vehicle_elev, under_vehicle_sem, timestr, timestep)
    
        # Action: (steer, speed, throttle, brake, gear)
        action = {
            'target_steer': target_steering,
            'real_steer': real_steering,
            'target_speed': target_speed,
            'real_speed': real_speed,
            'throttle': driver_inputs.m_throttle,
            'brake': driver_inputs.m_braking,
            'gear': gear
        }
        
        return state, action, True, time
    
    return None, None, False, last_log_time

def save_traj(trajectory_data, world_id, pos_id, timestr):
    """
    Save trajectory data as pickle file
    """
    # Create directory structure
    traj_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                           "./data/BenchMaps/sampled_maps/Expert", str(world_id))
    os.makedirs(traj_dir, exist_ok=True)
    
    # Generate timestamp
    filename = f"{pos_id}_{timestr}.pickle"
    filepath = os.path.join(traj_dir, filename)
    
    # Convert to DataFrame format
    df_data = []
    for timestep, (state, action) in enumerate(trajectory_data):
        row = {'timestep': timestep}
        row.update(state)
        row.update(action)
        df_data.append(row)
    
    with open(filepath, 'wb') as f:
        pickle.dump(df_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"Trajs saved: {filepath}")
    return filepath


def combine_rigid(m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
    rigid_sections = []
    max_dim = terrain_labels.shape[0]
    
    rigid_patches = defaultdict(set)
    for patch_file, i, j, center_pos in terrain_patches:
        label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
        if not texture_options[label]['is_deformable']:
            rigid_patches[label].add((i, j, center_pos))
            
    processed_patches = set()
    shapes = find_regular_shape(patch_size, max_dim)
    
    for label, patches in rigid_patches.items():
        patch_coords = {(i, j) for i, j, _ in patches}
        
        while patch_coords:
            best_shape, selected_patches = best_shape_fit(shapes, patch_size, patch_coords)
            
            if not best_shape or not selected_patches:
                break

            width, height = best_shape
            patches_width = (width - 1) // (patch_size - 1)
            patches_height = (height - 1) // (patch_size - 1)
            width_scaled = width * SCALE_FACTOR
            height_scaled = height * SCALE_FACTOR
            
            # Calculate bounds for this section
            min_i = min(i for i, j in selected_patches)
            min_j = min(j for i, j in selected_patches)
            max_i = max(i for i, j in selected_patches)
            max_j = max(j for i, j in selected_patches)
            
            # Find corner positions
            valid_corner_positions = []
            corner_coords = [(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)]
            for patch in patches:
                i, j, pos = patch
                if (i, j) in corner_coords and (i, j) in selected_patches:
                    valid_corner_positions.append(pos)
            
            # Calculate center position
            avg_x = sum(pos[0] for pos in valid_corner_positions) / len(valid_corner_positions)
            avg_y = sum(pos[1] for pos in valid_corner_positions) / len(valid_corner_positions)
            section_pos = chrono.ChVector3d(avg_x * SCALE_FACTOR, avg_y * SCALE_FACTOR, 0)
            
            # Check if all patches have the same properties
            if not selected_patches:
                raise ValueError("No patches selected for merging.")
            
            first_patch = next(iter(selected_patches))
            first_properties = property_dict[(first_patch[0], first_patch[1])]
            first_type = first_properties['terrain_type']
            first_texture = first_properties['texture_file']
            for patch in selected_patches:
                properties = property_dict[(patch[0], patch[1])]
                if properties['terrain_type'] != first_type:
                    raise ValueError(f"Terrain type mismatch: expected {first_type}, found {properties['terrain_type']}.")
                if properties['texture_file'] != first_texture:
                    raise ValueError(f"Texture file mismatch: expected {first_texture}, found {properties['texture_file']}.")
            
            # Create terrain section
            rigid_terrain = veh.RigidTerrain(m_system)
            patch_mat = chrono.ChContactMaterialNSC()
            patch_mat.SetFriction(properties['friction'])
            patch_mat.SetRestitution(properties['restitution'])
                
            if m_isFlat:
                patch = rigid_terrain.AddPatch(patch_mat, chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot), 
                                               width_scaled - SCALE_FACTOR, height_scaled - SCALE_FACTOR)
            else:
                start_i = min_i * (patch_size - 1)
                end_i = max_i * (patch_size - 1) + patch_size
                start_j = min_j * (patch_size - 1)
                end_j = max_j * (patch_size - 1) + patch_size
                
                file = terrain_patch_bmp(terrain_array,
                                       start_i, end_i,
                                       start_j, end_j,
                                       len(rigid_sections))
                                       
                patch = rigid_terrain.AddPatch(patch_mat,
                                            chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot),
                                            file,
                                            width_scaled - SCALE_FACTOR, height_scaled - SCALE_FACTOR,
                                            m_min_terrain_height,
                                            m_max_terrain_height)
            
            # Set texture
            patch.SetTexture(veh.GetDataFile(properties['texture_file']), patches_width, patches_height)
            rigid_terrain.Initialize()
            rigid_sections.append(rigid_terrain)
            
            # Update processed patches and remaining patches
            processed_patches.update(selected_patches)
            patch_coords -= selected_patches
    
    # Convert any remaining small patches individually
    for patch_file, i, j, center_pos in terrain_patches:
        if (i, j) not in processed_patches and not texture_options[terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]]['is_deformable']:
            properties = property_dict[(i, j)]
            patch_pos = chrono.ChVector3d(*center_pos) * SCALE_FACTOR
            
            rigid_terrain = veh.RigidTerrain(m_system)
            patch_mat = chrono.ChContactMaterialNSC()
            patch_mat.SetFriction(properties['friction'])
            patch_mat.SetRestitution(properties['restitution'])
            
            scaled_patch_size = patch_size * SCALE_FACTOR
            if m_isFlat:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                             scaled_patch_size - SCALE_FACTOR, scaled_patch_size - SCALE_FACTOR)
            else:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                             patch_file,
                                             scaled_patch_size - SCALE_FACTOR, scaled_patch_size - SCALE_FACTOR,
                                             m_min_terrain_height,
                                             m_max_terrain_height)
                                             
            patch.SetTexture(veh.GetDataFile(properties['texture_file']), patch_size, patch_size)
            rigid_terrain.Initialize()
            rigid_sections.append(rigid_terrain)
    
    return rigid_sections, property_dict, terrain_labels

def combine_deformation(m_system, terrain_patches, property_dict, texture_options, m_isFlat):
    type_to_label = {}
    deform_terrains = []
    
    for label, info in texture_options.items():
        type_to_label[info['terrain_type']] = label
    
    deformable_terrains = set(
        property_dict[(i, j)]['terrain_type']
        for _, i, j, _ in terrain_patches
        if property_dict[(i, j)]['is_deformable']
    )
    terrain_types = sorted(deformable_terrains)
    num_textures = len(terrain_types)
    bmp_width, bmp_height = terrain_array.shape
    
    if num_textures == 1:
        terrain_type = terrain_types[0]
        center_x, center_y = bmp_width // 2, bmp_height // 2
        chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
        section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * SCALE_FACTOR, 
                                        chrono_center_y - 0.5 * SCALE_FACTOR, 0)
            
        # Create terrain section
        deform_terrain = veh.SCMTerrain(m_system)
        
        # Set SCM parameters
        terrain_params = deformable_params(terrain_type)
        terrain_params.SetParameters(deform_terrain)
        
        # Enable bulldozing
        deform_terrain.EnableBulldozing(True)
        deform_terrain.SetBulldozingParameters(
            55,  # angle of friction for erosion
            1,   # displaced vs downward pressed material
            5,   # erosion refinements per timestep
            10   # concentric vertex selections
        )
        
        # Initialize terrain with regular shape dimensions
        deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
        deform_terrain.SetMeshWireframe(False)
        
        # Define size for deformable terrain
        width = 2 * m_terrain_length - 1 * SCALE_FACTOR
        height = 2 * m_terrain_width - 1 * SCALE_FACTOR
        
        # Create and set boundary
        aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
        deform_terrain.SetBoundary(aabb) 
        
        if m_isFlat:
            deform_terrain.Initialize(width, height, terrain_delta)
        else:
            deform_terrain.Initialize(
                terrain_path,
                width,
                height,
                m_min_terrain_height,
                m_max_terrain_height,
                terrain_delta
            )
        
        label = type_to_label[terrain_type]
        texture_file = texture_options[label]['texture_file']
        deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
        deform_terrains.append(deform_terrain)
            
    elif num_textures == 2:
        # Two textures: 1/2 for the first, 1/2 for the second
        split_height = bmp_height // 2

        for idx, terrain_type in enumerate(terrain_types):
            if idx == 0: # First texture
                start_y = 0
                end_y = split_height + 1
            else:  # Second texture
                start_y = split_height
                end_y = bmp_height
                
            section_height = end_y - start_y
            center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2     
            chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * SCALE_FACTOR, 
                                            chrono_center_y - 0.5 * SCALE_FACTOR, 0)
            
            # Create terrain section
            deform_terrain = veh.SCMTerrain(m_system)
            
            # Set SCM parameters
            terrain_params = deformable_params(terrain_type)
            terrain_params.SetParameters(deform_terrain)
            
            # Enable bulldozing
            deform_terrain.EnableBulldozing(True)
            deform_terrain.SetBulldozingParameters(
                55,  # angle of friction for erosion
                1,   # displaced vs downward pressed material
                5,   # erosion refinements per timestep
                10   # concentric vertex selections
            )
            
            # Initialize terrain with regular shape dimensions
            deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
            deform_terrain.SetMeshWireframe(False)
            
            # Define size for deformable terrain
            width = 2 * m_terrain_length - 1 * SCALE_FACTOR
            height = (section_height - 1) * (2 * m_terrain_width / bmp_height)
            
            # Create and set boundary
            aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
            deform_terrain.SetBoundary(aabb)  
            
            if m_isFlat:
                deform_terrain.Initialize(
                    width,
                    height,
                    terrain_delta
                )
            else:
                file = terrain_patch_bmp(terrain_array, start_y, end_y, 0, bmp_width, idx)
                deform_terrain.Initialize(
                    file,
                    width,
                    height,
                    m_min_terrain_height,
                    m_max_terrain_height,
                    terrain_delta
                )
            
            label = type_to_label[terrain_type]
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
            deform_terrains.append(deform_terrain)
            
    elif num_textures == 3:
        split_1 = bmp_height // 3
        
        for idx, terrain_type in enumerate(terrain_types):
            if idx == 0:  # Top texture
                start_y = 0
                end_y = split_1 + 1
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2

            elif idx == 1:  # Middle texture
                start_y = split_1
                end_y = split_1 * 2 + 1
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2

            else:  # Bottom texture
                start_y = split_1 * 2
                end_y = bmp_height
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2 - 0.5
            
            chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * SCALE_FACTOR, 
                                            chrono_center_y - 1 * SCALE_FACTOR, 0)
            
            # Create terrain section
            deform_terrain = veh.SCMTerrain(m_system)
            
            # Set SCM parameters
            terrain_params = deformable_params(terrain_type)
            terrain_params.SetParameters(deform_terrain)
            
            # Enable bulldozing
            deform_terrain.EnableBulldozing(True)
            deform_terrain.SetBulldozingParameters(
                55,  # angle of friction for erosion
                1,   # displaced vs downward pressed material
                5,   # erosion refinements per timestep
                10   # concentric vertex selections
            )
            
            # Initialize terrain with regular shape dimensions
            deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
            deform_terrain.SetMeshWireframe(False)
            
            width = 2 * m_terrain_length - 1 * SCALE_FACTOR
            height = (section_height - 1) * (2 * m_terrain_width / bmp_height)

            # Create and set boundary
            aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
            deform_terrain.SetBoundary(aabb)
                
            if m_isFlat:
                deform_terrain.Initialize(width, height, terrain_delta)
            else:
                file = terrain_patch_bmp(terrain_array, start_y, end_y, 0, bmp_width, idx)
                deform_terrain.Initialize(
                    file,
                    width,
                    height,
                    m_min_terrain_height,
                    m_max_terrain_height,
                    terrain_delta
                )
                
            label = type_to_label[terrain_type]
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
            deform_terrains.append(deform_terrain)
            
    return deform_terrains

def mixed_terrain(m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
    deformable_sections = []
    max_dim = terrain_labels.shape[0]
    
    deformable_patches = defaultdict(set)
    for patch_file, i, j, center_pos in terrain_patches:
        label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
        if texture_options[label]['is_deformable']:
            deformable_patches[label].add((i, j, center_pos))
            
    processed_patches = set()
    shapes = find_regular_shape(patch_size, max_dim)
    
    for label, patches in deformable_patches.items():
        patch_coords = {(i, j) for i, j, _ in patches}
        best_shape, selected_patches = best_shape_fit(shapes, patch_size, patch_coords)
        
        if not best_shape or not selected_patches:
            continue

        width, height = best_shape
        patches_width = (width - 1) // (patch_size - 1)
        patches_height = (height - 1) // (patch_size - 1)
        
        # Create deformable terrain for this shape
        deform_terrain = veh.SCMTerrain(m_system)
        terrain_type = texture_options[label]['terrain_type']
        terrain_params = deformable_params(terrain_type)
        terrain_params.SetParameters(deform_terrain)
        
        # Enable bulldozing
        deform_terrain.EnableBulldozing(True)
        deform_terrain.SetBulldozingParameters(
            55,  # angle of friction for erosion
            1,   # displaced vs downward pressed material
            5,   # erosion refinements per timestep
            10   # concentric vertex selections
        )
        
        # Calculate center in BMP coordinates
        min_i = min(i for i, j in selected_patches)
        min_j = min(j for i, j in selected_patches)
        max_i = max(i for i, j in selected_patches)
        max_j = max(j for i, j in selected_patches)
        
        valid_corner_positions = []
        corner_coords = [(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)]
        for patch in patches:
            i, j, pos = patch
            if (i, j) in corner_coords and (i, j) in selected_patches:
                valid_corner_positions.append(pos)
        
        # Calculate average center position
        avg_x = sum(pos[0] for pos in valid_corner_positions) / len(valid_corner_positions)
        avg_y = sum(pos[1] for pos in valid_corner_positions) / len(valid_corner_positions)
        section_pos = chrono.ChVector3d(avg_x * SCALE_FACTOR, avg_y * SCALE_FACTOR, 0)
        
        # Initialize terrain section
        deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
        deform_terrain.SetMeshWireframe(False)
        
        width_scaled = width * SCALE_FACTOR
        height_scaled = height * SCALE_FACTOR
        
        # Create and set boundary
        aabb = chrono.ChAABB(chrono.ChVector3d(-width_scaled/2, -height_scaled/2, 0), chrono.ChVector3d(width_scaled/2, height_scaled/2, 0))
        deform_terrain.SetBoundary(aabb)  
        
        if m_isFlat:
            deform_terrain.Initialize(width_scaled - SCALE_FACTOR, height_scaled - SCALE_FACTOR, terrain_delta)
        else:
            start_i = min_i * (patch_size - 1)
            end_i = max_i * (patch_size - 1) + patch_size
            start_j = min_j * (patch_size - 1)
            end_j = max_j * (patch_size - 1) + patch_size
            file = terrain_patch_bmp(terrain_array, 
                                    start_i, end_i,
                                    start_j, end_j,
                                    len(deformable_sections))
            deform_terrain.Initialize(
                file,
                width_scaled - SCALE_FACTOR, height_scaled - SCALE_FACTOR,
                m_min_terrain_height,
                m_max_terrain_height,
                terrain_delta
            )
        
        # Set texture
        texture_file = texture_options[label]['texture_file']
        deform_terrain.SetTexture(veh.GetDataFile(texture_file), patches_width, patches_height)
        deformable_sections.append(deform_terrain)
        processed_patches.update(selected_patches)
            
    # Convert remaining deformable patches to first rigid texture
    first_rigid_label = min(label for label, info in texture_options.items() if not info['is_deformable'])
    first_rigid_info = next(info for info in textures if info['terrain_type'] == texture_options[first_rigid_label]['terrain_type'])
    
    updated_property_dict = property_dict.copy()
    for patch_file, i, j, center_pos in terrain_patches:
        if (i, j) not in processed_patches and texture_options[terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]]['is_deformable']:
            updated_property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': first_rigid_info['terrain_type'],
                'texture_file': texture_options[first_rigid_label]['texture_file'],
                'friction': first_rigid_info['friction'],
                'restitution': first_rigid_info.get('restitution', 0.01)
            }
            terrain_labels[i * (patch_size - 1):(i + 1) * (patch_size - 1), 
                         j * (patch_size - 1):(j + 1) * (patch_size - 1)] = first_rigid_label

    return deformable_sections, updated_property_dict, terrain_labels

def load_texture_config():
    property_dict = {}
    terrain_patches = []
    
    labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "./data/BenchMaps/sampled_maps/Configs/Final", f"labels{world_id}_*.npy")
    matched_labels = glob.glob(labels_path)
    labels_path = matched_labels[0]
    terrain_labels = np.load(labels_path)
    
    texture_options = {}
    terrain_type_to_label = {
        'clay': 0, 'concrete': 1, 'dirt': 2, 'grass': 3, 
        'gravel': 4, 'rock': 5, 'wood': 6,
        'mud': 7, 'sand': 8, 'snow': 9
    }
    
    # Process each texture configuration
    for texture_info in textures:
        i, j = texture_info['index']
        terrain_type = texture_info['terrain_type']
        label = terrain_type_to_label[terrain_type]
        
        center_pos = (
            texture_info['center_position']['x'],
            texture_info['center_position']['y'],
            texture_info['center_position']['z']
        )
        patch_filename = f"patch_{i}_{j}.bmp"
        terrain_patches.append((patch_filename, i, j, center_pos))
        
        # Update texture options
        texture_options[label] = {
            'texture_file': texture_info['texture_file'],
            'terrain_type': terrain_type,
            'is_deformable': texture_info['is_deformable']
        }
        
        # Update property dictionary
        if texture_info['is_deformable']:
            property_dict[(i, j)] = {
                'is_deformable': True,
                'terrain_type': terrain_type,
                'texture_file': texture_info['texture_file']
            }
        else:
            property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': terrain_type,
                'texture_file': texture_info['texture_file'],
                'friction': texture_info['friction'],
                'restitution': texture_info.get('restitution', 0.01)
            }
        
    return property_dict, terrain_labels, texture_options, terrain_patches    

def add_obstacles(m_system, m_isFlat=False):
    m_assets = SimulationAssets(m_system, m_terrain_length * 1.8, m_terrain_width * 1.8, SCALE_FACTOR,
                                high_res_data, m_min_terrain_height, m_max_terrain_height, m_isFlat)
    
    # Add rocks
    for rock_info in config['obstacles']['rocks']:
        rock_scale = rock_info['scale'] * SCALE_FACTOR
        rock_pos = chrono.ChVector3d(rock_info['position']['x'] * SCALE_FACTOR,
                                  rock_info['position']['y'] * SCALE_FACTOR,
                                  rock_info['position']['z'] * SCALE_FACTOR)
        
        rock = Asset(visual_shape_path="sensor/offroad/rock.obj",
                    scale=rock_scale,
                    bounding_box=chrono.ChVector3d(4.4, 4.4, 3.8))
        
        asset_body = rock.Copy()
        asset_body.UpdateAssetPosition(rock_pos, chrono.ChQuaterniond(1, 0, 0, 0))
        m_system.Add(asset_body.body)
    
    # Add trees
    for tree_info in config['obstacles']['trees']:
        tree_pos = chrono.ChVector3d(tree_info['position']['x'] * SCALE_FACTOR,
                                  tree_info['position']['y'] * SCALE_FACTOR,
                                  tree_info['position']['z'] * SCALE_FACTOR)
        
        tree = Asset(visual_shape_path="sensor/offroad/tree.obj",
                    scale=1.0 * SCALE_FACTOR,
                    bounding_box=chrono.ChVector3d(1.0, 1.0, 5.0))
        
        asset_body = tree.Copy()
        asset_body.UpdateAssetPosition(tree_pos, chrono.ChQuaterniond(1, 0, 0, 0))
        m_system.Add(asset_body.body)
    
    return m_assets

def run_simulation(render=False, use_gui=False, m_isFlat = False, is_rigid=False, is_deformable=False, obstacles_flag=False):
    if use_gui and not render:
        raise ValueError("If use_gui is True, render must also be True. GUI requires rendering.")
    
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    # Set thread counts based on available CPUs
    num_procs = multiprocessing.cpu_count()
    num_threads_chrono = min(8, num_procs)
    num_threads_collision = min(8, num_procs)
    num_threads_eigen = 1
    m_system.SetNumThreads(num_threads_chrono, num_threads_collision, num_threads_eigen)
        
    # Visualization frequencies
    m_vis_freq = 100.0  # Hz
    m_vis_dur = 1.0 / m_vis_freq
    last_vis_time = 0.0
    
    # Control paras
    last_control_time = 0.0 
    sine_steering = SineSteer(steering_freq)
    sine_speed = SineSpeed(speed_freq)  
    target_speed = 0.0
    target_steering = 0.0
    
    # Trajectory logging setup
    last_log_time = None
    traj_data = []
    timestr = '{0:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    
    # # Replan frequency
    # last_replan_time = 0
    # replan_interval = 0.1
    
    # Vehicle setup
    m_vehicle = veh.HMMWV_Reduced(m_system)
    m_vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
    m_vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    m_vehicle.SetChassisFixed(False)
    m_vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP) # This offers higher max torques 
    m_vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    m_vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    m_vehicle.SetTireType(veh.TireModelType_RIGID)
    m_vehicle.SetTireStepSize(m_step_size)
    m_vehicle.SetInitFwdVel(0.0)
    m_initLoc, m_initRot, m_initYaw = initialize_vw_pos(m_vehicle, start_pos, goal_pos, m_isFlat)
    m_goal = set_goal(m_system, goal_pos, m_isFlat)
    m_vehicle.Initialize()

    m_vehicle.LockAxleDifferential(0, True)    
    m_vehicle.LockAxleDifferential(1, True)
    m_vehicle.LockCentralDifferential(0, True)
    m_vehicle.GetVehicle().EnableRealtime(False)
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_chassis_body = m_vehicle.GetChassisBody()
    m_transmission = m_vehicle.GetVehicle().GetTransmission()
    
    # Setup texture configurations
    property_dict, terrain_labels, texture_options, terrain_patches = load_texture_config()
    
    # Load high resolution terrain labels
    high_res_factor = high_res_dim_x // terrain_labels.shape[1]
    high_res_terrain_labels = np.zeros((terrain_labels.shape[0] * high_res_factor, 
                                        terrain_labels.shape[1] * high_res_factor), dtype=np.int32)
    for i in range(terrain_labels.shape[0]):
        for j in range(terrain_labels.shape[1]):
            label_value = terrain_labels[i, j]
            i_start = i * high_res_factor
            j_start = j * high_res_factor
            high_res_terrain_labels[i_start:i_start+high_res_factor, 
                                    j_start:j_start+high_res_factor] = label_value
    
    if terrain_type == 'rigid':
        original_labels = terrain_labels.copy()
        rigid_terrains, property_dict, _ = combine_rigid(
            m_system, terrain_patches, terrain_labels.copy(), property_dict,
            texture_options, patch_size, m_isFlat
        )
        terrain_labels = original_labels
              
    elif terrain_type == 'deformable':
        original_labels = terrain_labels.copy()
        deform_terrains = combine_deformation(m_system, terrain_patches, property_dict, texture_options, m_isFlat)
        terrain_labels = original_labels
             
    else: 
        original_labels = terrain_labels.copy()
        deform_terrains, property_dict, _ = mixed_terrain(
            m_system, terrain_patches, terrain_labels.copy(), property_dict,
            texture_options, patch_size, m_isFlat
        )
        rigid_terrains, property_dict, _ = combine_rigid(
            m_system, terrain_patches, original_labels, property_dict,
            texture_options, patch_size, m_isFlat
        )
        terrain_labels = original_labels
        
    if obstacles_flag:
        add_obstacles(m_system, m_isFlat=m_isFlat)
    
    if is_deformable:
        for deform_terrain in deform_terrains:
            for axle in m_vehicle.GetVehicle().GetAxles():
                deform_terrain.AddMovingPatch(
                    axle.m_wheels[0].GetSpindle(), 
                    chrono.VNULL, 
                    chrono.ChVector3d(1.0 * SCALE_FACTOR, 0.6 * SCALE_FACTOR, 1.0 * SCALE_FACTOR)
                )
                deform_terrain.AddMovingPatch(
                    axle.m_wheels[1].GetSpindle(), 
                    chrono.VNULL, 
                    chrono.ChVector3d(1.0 * SCALE_FACTOR, 0.6 * SCALE_FACTOR, 1.0 * SCALE_FACTOR)
                )
            deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    
    # Visualization
    if render:
        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle('vws in the wild')
        vis.SetWindowSize(3840, 2160)
        trackPoint = chrono.ChVector3d(-1.0, 0.0, 1.75)
        vis.SetChaseCamera(trackPoint, 6.0, 0.5)
        vis.Initialize()
        vis.AddLightDirectional()
        vis.AddSkyBox()
        vis.AttachVehicle(m_vehicle.GetVehicle())
        vis.EnableStats(True)
        
    # Set the driver
    if use_gui:
        # GUI-based interactive driver system
        m_driver = veh.ChInteractiveDriverIRR(vis)
        m_driver.SetSteeringDelta(0.1) # Control sensitivity
        m_driver.SetThrottleDelta(0.02)
        m_driver.SetBrakingDelta(0.06)
        m_driver.Initialize()
    else:
        # Automatic driver
        m_driver = veh.ChDriver(m_vehicle.GetVehicle())
        
    m_driver_inputs = m_driver.GetInputs()
    # Set PID controller for speed
    m_speedController = veh.ChSpeedController()
    m_speedController.Reset(m_vehicle.GetRefFrame())
    m_speedController.SetGains(1.0, 0.0, 0.0)
    # Initialize the custom PID controller for steering
    m_steeringController = PIDController(kp=1.0, ki=0.0, kd=0.0)
    
    # # A* path planning in chrono
    # local_goal_idx = 0
    # chrono_path = astar_path(obs_path, start_pos, goal_pos)
    
    # Continuous speed
    speed = 4.0 
    start_time = m_system.GetChTime()
    
    # Check if the vehicle is stuck
    last_position = None
    stuck_counter = 0
    STUCK_DISTANCE = 0.01
    STUCK_TIME = 10.0
    
    # # Visualize cropped elev and semantic maps
    # cv2.namedWindow("Elevation", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Semantic", cv2.WINDOW_NORMAL)
    
    # Initialize manual driver input by G29
    m_manual_input = ManualInput()
    threading.Thread(target=m_manual_input.read_loop, daemon=True).start()
    
    while True:
        time = m_system.GetChTime()
        
        if render:
            if not vis.Run():
                break
            
        # Draw at low frequency
        if render:
            if last_vis_time==0 or (time - last_vis_time) > m_vis_dur:
                vis.BeginScene()
                vis.Render()
                vis.EndScene()
                last_vis_time = time
            
        m_vehicle_pos = m_vehicle.GetVehicle().GetPos() #Global coordinate
        m_vector_to_goal = m_goal - m_vehicle_pos 
        
        # Collect trajs
        state, action, logged, last_log_time = collect_traj(
            m_vehicle, m_driver_inputs, time, wait_time_before_log, last_log_time, 
            log_dur, timestr, target_speed, target_steering
        )
        if logged:
            traj_data.append((state, action))
        
        # # Visualize cropped elev and semantic maps
        # under_elev, _ = get_cropped_elev(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 128, 5)
        # under_sem      = get_cropped_sem(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 128)

        # # Normalize elevation
        # if under_elev.max() > under_elev.min():
        #     e8 = ((under_elev - under_elev.min()) / under_elev.ptp() * 255).astype(np.uint8)
        # else:
        #     e8 = np.zeros_like(under_elev, dtype=np.uint8)
        # under_sem = cv2.cvtColor(under_sem, cv2.COLOR_RGB2BGR)

        # # Show
        # cv2.imshow("Elevation", e8)
        # cv2.imshow("Semantic",   under_sem)
        # if cv2.waitKey(1) == 27:  # ESC to quit early
        #     break
            
        if use_gui:
            if last_control_time == 0 or (time - last_control_time) >= m_control_dur: 
                m_driver_inputs.m_steering = m_manual_input.steering_n
                m_driver_inputs.m_throttle = m_manual_input.accel_n
                m_driver_inputs.m_braking = m_manual_input.brakes_n
                m_driver_inputs.m_direction = m_manual_input.direction_n
                m_transmission.SetGear(m_manual_input.direction_n)   
                
                out_throttle = m_speedController.Advance(
                    m_vehicle.GetVehicle().GetRefFrame(), 
                    speed * m_driver_inputs.m_direction, 
                    time, 
                    m_control_dur
                )
                out_throttle = np.clip(out_throttle, -1, 1)
                
                # Split into throttle and braking
                if out_throttle > 0:
                    throttle, braking = out_throttle, 0.0
                else:
                    throttle, braking = 0.0, -out_throttle
                
                if m_manual_input.direction_n < 0:
                    m_driver_inputs.m_throttle = np.clip(m_driver_inputs.m_throttle, 0, braking)
                else:
                    m_driver_inputs.m_throttle = np.clip(m_driver_inputs.m_throttle, 0, throttle)
                    
                last_control_time = time
                
        else:       
            if last_control_time == 0 or (time - last_control_time) >= m_control_dur: 
                euler_angles = m_vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ() #Global coordinate
                vehicle_heading = euler_angles.z
            
                # if time - last_replan_time >= replan_interval:
                #     print("Replanning path...")
                #     new_path = astar_replan(
                #         obs_path,
                #         (m_vehicle_pos.x, m_vehicle_pos.y),
                #         (m_goal.x, m_goal.y)
                #     )
                    
                #     if new_path is not None:
                #         chrono_path = new_path
                #         local_goal_idx = 0
                #         print("Path replanned successfully")
                #     else:
                #         print("Failed to replan path!")
                    
                #     last_replan_time = time
                
                # local_goal_idx, local_goal = find_local_goal(
                #     (m_vehicle_pos.x, m_vehicle_pos.y), 
                #     vehicle_heading, 
                #     chrono_path, 
                #     local_goal_idx
                # )
                
                # goal_heading = np.arctan2(local_goal[1] - m_vehicle_pos.y, local_goal[0] - m_vehicle_pos.x)
                # goal_heading = np.arctan2(m_goal.y - m_vehicle_pos.y, m_goal.x - m_vehicle_pos.x)
                # heading_error = (goal_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
                
                # Generate sine wave inputs
                target_steering = sine_steering.advance(time)
                target_speed = sine_speed.advance(time)
                
                #PID controller for steering
                # steering = -m_steeringController.compute(heading_error, m_control_dur)
                m_driver_inputs.m_steering = np.clip(target_steering, 
                                                     m_driver_inputs.m_steering - m_steeringDelta, 
                                                     m_driver_inputs.m_steering + m_steeringDelta)
                target_steering = m_driver_inputs.m_steering
                
                # Desired throttle/braking value
                out_throttle = m_speedController.Advance(m_vehicle.GetVehicle().GetRefFrame(), target_speed, time, m_control_dur)
                out_throttle = np.clip(out_throttle, -1, 1)
                if out_throttle > 0:
                    m_driver_inputs.m_braking = 0
                    m_driver_inputs.m_throttle = out_throttle
                else:
                    m_driver_inputs.m_braking = -out_throttle
                    m_driver_inputs.m_throttle = 0
                    
                last_control_time = time
        
        current_position = (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z)
        
        if last_position:
            position_change = np.sqrt(
                (current_position[0] - last_position[0])**2 +
                (current_position[1] - last_position[1])**2 +
                (current_position[2] - last_position[2])**2
            )
            
            if position_change < STUCK_DISTANCE:
                stuck_counter += m_step_size
            else:
                stuck_counter = 0
                
            if stuck_counter >= STUCK_TIME:
                print('--------------------------------------------------------------')
                print('Vehicle stuck!')
                print(f'Stuck time: {stuck_counter:.2f} seconds')
                print(f'Position change: {position_change:.3f} m')
                print(f'Initial position: {m_initLoc}')
                print(f'Current position: {m_vehicle_pos}')
                print(f'Goal position: {m_goal}')
                print(f'Distance to goal: {m_vector_to_goal.Length():.2f} m')
                print('--------------------------------------------------------------')
                
                if traj_data:
                    save_traj(traj_data, world_id, pos_id, timestr)
                
                if render:
                    vis.Quit()

                return time - start_time, False
        
        last_position = current_position
        
        if m_vector_to_goal.Length() < 8 * SCALE_FACTOR:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print('--------------------------------------------------------------')
            
            if traj_data:
                save_traj(traj_data, world_id, pos_id, timestr)
                    
            if render:
                vis.Quit()
                
            return time - start_time, True
        
        if (abs(m_vehicle_pos.x) >= m_terrain_length * 0.97 or abs(m_vehicle_pos.y) >= m_terrain_width * 0.97):
            print('--------------------------------------------------------------')
            print('Vehicle out of bounds!')
            print(f'Vehicle position: ({m_vehicle_pos.x:.2f}, {m_vehicle_pos.y:.2f})')
            print(f'Terrain bounds: X = ±{m_terrain_length:.2f}, Y = ±{m_terrain_width:.2f}')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print(f'Distance to goal: {m_vector_to_goal.Length():.2f} m')
            print('--------------------------------------------------------------')
            
            if traj_data:
                save_traj(traj_data, world_id, pos_id, timestr)
            
            if render:
                vis.Quit()
                
            return time - start_time, False
        
        if m_system.GetChTime() > m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', m_initLoc)
            dist = m_vector_to_goal.Length()
            print('Final position of vw: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            
            if traj_data:
                save_traj(traj_data, world_id, pos_id, timestr)
            
            if render:
                vis.Quit()
                
            return time - start_time, False
        
        if is_rigid:
            # print("Rigid terrain:", len(rigid_terrains))
            for rigid_terrain in rigid_terrains:
                rigid_terrain.Synchronize(time)
                m_vehicle.Synchronize(time, m_driver_inputs, rigid_terrain)
                rigid_terrain.Advance(m_step_size)
            
        if is_deformable:
            # print("Deform terrain:", len(deform_terrains))
            for deform_terrain in deform_terrains:
                deform_terrain.Synchronize(time)
                m_vehicle.Synchronize(time, m_driver_inputs, deform_terrain)
                deform_terrain.Advance(m_step_size)
        
        m_driver.Advance(m_step_size)
        m_vehicle.Advance(m_step_size)
        
        if render:
            vis.Synchronize(time, m_driver_inputs)
            vis.Advance(m_step_size)
        
        m_system.DoStepDynamics(m_step_size)

    return time - start_time, False
    
if __name__ == '__main__':
    # Terrain parameters
    SetChronoDataDirectories()
    
    # Simulation params
    SCALE_FACTOR = 1.0
    m_max_time = 60
    m_step_size = 5e-3 # sim: 200Hz
    wait_time_before_log = 1.0 # robot has to be dropped first
    
    # Control and dynamics frequency
    m_control_freq = 20.0 # Hz
    m_control_dur = 1.0 / m_control_freq
    
    # Control oscillation rate
    steering_freq = 0.5
    speed_freq = 2.5
    m_steeringDelta = 0.025 # At max the steering can change by 0.025 in 0.05 seconds
    
    # Log data
    log_freq = 20.0 # Hz
    log_dur = 1.0 / log_freq
    
    # Path Plan Paras
    inflation_radius = int(4.0 * SCALE_FACTOR)
    look_ahead_distance = 10.0 * SCALE_FACTOR
    
    # Parse command line arguments
    processed_args = []
    for arg in sys.argv[1:]: 
        if '=' in arg and not arg.startswith('-'):
            key, value = arg.split('=', 1)
            processed_args.append(f"--{key}")
            processed_args.append(value)
        else:
            processed_args.append(arg)
    parser = argparse.ArgumentParser(description='Input world ID')
    parser.add_argument('--world_id', type=int, default=1, help='World ID (1-100, default: 1)')
    args = parser.parse_args(processed_args)
    if args.world_id < 1 or args.world_id > 100:
        raise ValueError("world_id must be between 1 and 100")

    world_id = args.world_id
    print(f"Processing World {world_id}/{100}")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/Final", f"config{world_id}_*.yaml")
    matched_file = glob.glob(config_path)
    config_path = matched_file[0]
    print(f"Using config file: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    m_terrain_length = config['terrain']['length'] * SCALE_FACTOR
    m_terrain_width = config['terrain']['width'] * SCALE_FACTOR
    m_min_terrain_height = config['terrain']['min_height'] * SCALE_FACTOR
    m_max_terrain_height = config['terrain']['max_height'] * SCALE_FACTOR
    difficulty = config['terrain']['difficulty']
    m_isFlat = config['terrain']['is_flat']
    positions = config['positions']
    terrain_type = config['terrain_type']
    obstacle_flag = config['obstacles_flag']
    obstacle_density = config['obstacle_density']
    textures = config['textures']
    terrain_delta = 0.1 * SCALE_FACTOR # mesh resolution for SCM terrain
    patch_size = 9
    
    # Load terrain bitmap
    terrain_file = f"{world_id}.bmp"
    terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "./data/BenchMaps/sampled_maps/Worlds", terrain_file)
    
    terrain_image = Image.open(terrain_path)
    terrain_array = np.array(terrain_image)
    bmp_dim_y, bmp_dim_x = terrain_array.shape 
    if (bmp_dim_y, bmp_dim_x) != (129, 129):
        raise ValueError("Check terrain file and dimensions")

    # Load high resolution terrain data
    high_res_file = f"height{world_id}_*.npy"
    high_res_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/Final", high_res_file)
    actual_file_path = glob.glob(high_res_path)[0]
    high_res_data = np.load(actual_file_path)
    high_res_data = np.flip(high_res_data, axis=1)
    high_res_data = np.rot90(high_res_data, k=1, axes=(1, 0))
    high_res_data = np.rot90(high_res_data, k=1, axes=(1, 0))
    high_res_dim_y, high_res_dim_x = high_res_data.shape
    if (high_res_dim_y, high_res_dim_x) != (1291, 1291):
        raise ValueError("Check high resolution height map dimensions!")
    
    # Load high resolution semantic data
    high_sem_file = f"sem{world_id}_*.png"
    high_sem_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/Final", high_sem_file)
    actual_sem_path = glob.glob(high_sem_path)[0]
    actual_sem_img = Image.open(actual_sem_path)
    high_sem_data = np.array(actual_sem_img)
    high_sem_data = np.rot90(high_sem_data, k=1, axes=(1,0))
    high_sem_dim_y, high_sem_dim_x = high_sem_data.shape[:2]
    if (high_sem_dim_y, high_sem_dim_x) != (1291, 1291):
        raise ValueError("Check high resolution semantic map dimensions!")
    
    # Load obstacle map
    obs_file = f"obs{world_id}_{difficulty}.bmp"
    obs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/Final", obs_file)
    
    pos_id = random.randint(0, len(positions) - 1)
    # Start and goal positions
    selected_pair = positions[pos_id]
    start_pos = selected_pair['start']
    goal_pos = selected_pair['goal']
    
    if terrain_type == 'rigid':
        is_rigid = True
        is_deformable = False
    elif terrain_type == 'deformable':
        is_rigid = False
        is_deformable = True
    else:
        is_rigid = True
        is_deformable = True

    time_to_goal, success, = run_simulation(render=True, use_gui=True, m_isFlat=m_isFlat,
                                            is_rigid=is_rigid, is_deformable=is_deformable, 
                                            obstacles_flag=False)
        
    print("Trajectories Process Complete!")
