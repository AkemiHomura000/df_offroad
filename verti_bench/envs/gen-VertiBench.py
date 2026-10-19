import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import math

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens
from verti_bench.envs.utils.asset_utils import *

from PIL import Image, ImageDraw
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import copy
import yaml
import logging
import cv2
import multiprocessing
import glob
import re

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from collections import defaultdict

class GeometryDiff:
    def __init__(self):
        self.difficulty_levels = {
            'low': 0.3,    # 30% of original height
            'mid': 0.6,    # 60% of original height
            'high': 1.0    # original scale
        }
        
    def get_height_range(self, difficulty):
        if difficulty not in self.difficulty_levels:
            raise ValueError(f"Invalid difficulty level. Must be one of {list(self.difficulty_levels.keys())}")
            
        scale = self.difficulty_levels[difficulty]
        min_height = 0
        max_height = 16 * scale
        
        return min_height, max_height

class RigidProperties:
    def __init__(self):
        # Friction coefficients: (mean, standard_deviation)
        self.friction_properties = {
            'clay': (0.6, 0.05),    
            'concrete': (0.85, 0.08), 
            'dirt': (0.6, 0.1),     
            'grass': (0.36, 0.07),  
            'gravel': (0.55, 0.1),   
            'rock': (0.65, 0.08),   
            'wood': (0.45, 0.06)    
        }
        
    def sample_properties(self, terrain_type):
        """
        Sample friction and restitution values from Gaussian distributions
        for the given terrain type.
        """
        # Sample friction coefficient
        friction_mean, friction_std = self.friction_properties[terrain_type]
        friction = np.random.normal(friction_mean, friction_std)
        friction = np.clip(friction, 0.1, 1.0)
        restitution = 0.01
        
        return friction, restitution

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

def generate_circle_pairs(center, radius, num_pairs, terrain_half_length, terrain_half_width):
    """
    Generate fixed (start, goal) pairs on a circle.
    
    Parameters:
    - center: Tuple (x, y, z) representing the circle center.
    - radius: Radius of the circle.
    - num_pairs: Number of (start, goal) pairs to generate.

    Returns:
    - pairs: List of (start, goal) pairs where each pair is a tuple ((sx, sy, sz), (gx, gy, gz)).
    """
    pairs = []
    angle_step = 2 * np.pi / num_pairs
    for i in range(num_pairs):
        # Start position
        start_angle = i * angle_step
        sx = center[0] + radius * np.cos(start_angle)
        sy = center[1] + radius * np.sin(start_angle)
        sz = center[2] 

        # Goal position (directly opposite)
        goal_angle = start_angle + np.pi 
        gx = center[0] + radius * np.cos(goal_angle)
        gy = center[1] + radius * np.sin(goal_angle)
        gz = center[2] 

        if (abs(sx) <= terrain_half_length and abs(sy) <= terrain_half_width and
            abs(gx) <= terrain_half_length and abs(gy) <= terrain_half_width):
            pairs.append(((sx, sy, sz), (gx, gy, gz)))
    return pairs

def transform_to_bmp(chrono_positions):
    """
    Pychrono coordinate: X (Forward), Y (Left), Z (Up), Origin (center)
    BMP coordinate: X (Right), Y (Down), Origin (top-left) 
    
    Return:
    - bmp_positions[i][0] is the column in the bitmap
    - bmp_positions[i][1] is the row in the bitmap
    """
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
        vehicle_x = pos[0]  # PyChrono X (Forward)
        vehicle_y = -pos[1]  # PyChrono Y (Left)
        pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])
        
        # Transform to BMP coordinates
        pos_bmp = np.dot(T, pos_chrono)
        bmp_positions.append((pos_bmp[0], pos_bmp[1]))
    
    return bmp_positions

def transform_to_chrono(bmp_positions):
    """
    Return:
    - x corresponds to the PyChrono X (Forward) axis
    - y corresponds to the PyChrono Y (Left) axis
    """
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
        # Transform back to PyChrono coordinates
        pos_bmp = np.array([pos[0], pos[1], 1])
        pos_chrono = np.dot(T_inv, pos_bmp)

        # Adjust to PyChrono coordinate system
        x = (pos_chrono[0] - m_terrain_length) 
        y = -(pos_chrono[1] - m_terrain_width) 
        chrono_positions.append((x, y))

    return chrono_positions

def transform_to_high_res(chrono_positions, height_array=None):
    """
    Transform PyChrono vehicle positions to high-resolution BMP/Labels coordinates.

    Args:
        chrono_positions: List of (x, y) positions in PyChrono coordinates.
        high_res_terrain_array: The 1291x1291 heightmap array.
        m_terrain_length: Half-length of the terrain in PyChrono (same as low-res).
        m_terrain_width: Half-width of the terrain in PyChrono (same as low-res).

    Returns:
        # List of (column, row) positions in the high-resolution BMP/Labels.
    """
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
        # Adjust PyChrono coordinates
        vehicle_x = pos[0]  # PyChrono X (Forward)
        vehicle_y = -pos[1]  # PyChrono Y (Left)
        pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])
        
        # Transform to BMP coordinates
        pos_bmp = np.dot(T, pos_chrono)
        bmp_positions.append((pos_bmp[0], pos_bmp[1]))
    
    return bmp_positions

def initialize_vw_pos(m_vehicle, start_pos, goal_pos, m_isFlat):
    """
    Initialize the vehicle position and rotation 
    based on the start and goal positions.
    """
    if m_isFlat:
        start_height = 0
    else:
        pos_bmp = transform_to_high_res([start_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1))) 
        start_height = high_res_data[pos_bmp_y, pos_bmp_x]

    start_pos = (start_pos[0], start_pos[1], start_height + start_pos[2])
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    start_yaw = np.arctan2(dy, dx)
    m_initLoc = chrono.ChVector3d(*start_pos)
    m_initRot = chrono.QuatFromAngleZ(start_yaw)
    m_vehicle.SetInitPosition(chrono.ChCoordsysd(m_initLoc, m_initRot))
    return m_initLoc, m_initRot, start_yaw

def set_goal(m_system, goal_pos, m_isFlat):
    """
    Set the goal position for the vehicle.
    """
    if m_isFlat:
        goal_height = 0
    else:
        pos_bmp = transform_to_high_res([goal_pos])[0] 
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1)))
        goal_height = high_res_data[pos_bmp_y, pos_bmp_x]
    
    offset = 1.0
    goal_pos = (goal_pos[0], goal_pos[1], goal_height + goal_pos[2] + offset)
    m_goal = chrono.ChVector3d(*goal_pos)

    # Create goal sphere with visualization settings
    goal_contact_material = chrono.ChContactMaterialNSC()
    goal_body = chrono.ChBodyEasySphere(0.5, 1000, True, False, goal_contact_material)
    goal_body.SetPos(m_goal)
    goal_body.SetFixed(True)
    
    # Apply red visualization material
    goal_mat = chrono.ChVisualMaterial()
    goal_mat.SetAmbientColor(chrono.ChColor(1, 0, 0)) 
    goal_mat.SetDiffuseColor(chrono.ChColor(1, 0, 0))
    goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
    
    # Add the goal body to the system
    m_system.Add(goal_body)
    return m_goal

def get_cropped_map(m_vehicle, m_vehicle_pos, region_size, num_front_regions):
    """
    Create height maps of terrain under/in front of the vehicle 
    from the high-resolution height map.
    """
    bmp_dim_y, bmp_dim_x = high_res_data.shape  # height (rows), width (columns)
    pos_bmp = transform_to_high_res([m_vehicle_pos])[0]
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
    vehicle_heading_global = m_vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
    angle = np.degrees(vehicle_heading_global) % 360
    
    # Using tensor to accelerate the rotation process
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor_map = torch.tensor(shifted_map, device=device).unsqueeze(0).float()
    rotated_tensor = F.rotate(tensor_map, -angle)
    rotated_map = rotated_tensor.squeeze().cpu().numpy()
    rotated_map = np.fliplr(rotated_map)

    # Extract the part under the vehicle
    # Vehicle's x-forward direction becomes negative Y in BMP space
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

def get_current_label(m_vehicle, m_vehicle_pos, region_size, high_res_terrain_labels): 
    """
    Extract the terrain type labels beneath the vehicle from 
    the high-resolution terrain labels.
    """       
    bmp_dim_y, bmp_dim_x = high_res_terrain_labels.shape  # height (rows), width (columns)
    pos_bmp = transform_to_high_res([m_vehicle_pos], high_res_terrain_labels)[0]
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
    vehicle_heading_global = m_vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ().z
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

def find_similar_region(under_vehicle, front_regions, region_size):
    """
    Identify the terrain region in front of the vehicle that has the most 
    similar characteristics to the terrain currently underneath the vehicle.
    """
    under_mean = np.mean(under_vehicle)
    under_var = np.var(under_vehicle)
    region_stats = [(np.mean(region), np.var(region)) for region in front_regions]

    # Sort regions by similarity to under_vehicle
    # Primary criterion: mean similarity; Secondary criterion: variance similarity
    sorted_regions = sorted(
        enumerate(region_stats), 
        key=lambda stat: (abs(stat[1][0] - under_mean), abs(stat[1][1] - under_var))
    )
    # Select the best region (most similar)
    target_index, (target_mean, target_variance) = sorted_regions[0]
    
    # Calculate the relative x position of the target region
    center_index = len(front_regions) // 2
    relative_x = (target_index - center_index) * region_size + region_size / 2
    relative_y = region_size / 2

    return relative_x, relative_y

def get_patch_center_position(i, j, patch_size, terrain_length, terrain_width):
    """
    Calculate the center position of each small patch based on its index (i, j).
    """
    x_start = j * (patch_size - 1)
    y_start = i * (patch_size - 1)
    x_center = -((terrain_length - 1) / 2) + (x_start + ((patch_size - 1) / 2))
    y_center = ((terrain_width - 1) / 2) - (y_start + ((patch_size - 1) / 2))
    return x_center, y_center, 0

def divide_terrain_image(output_folder, patch_size=65):
    """
    Divide the terrain image into smaller patches and save them.
    Example: [
        ('patches/patch_0_0.bmp', 0, 0, (-32.0, 32.0, 0)),
        ('patches/patch_0_1.bmp', 0, 1, (32.0, 32.0, 0)),
    ]
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    #Save the image for debugging
    # os.makedirs(output_folder, exist_ok=True)
    bmp_dim_y, bmp_dim_x = terrain_array.shape

    # Get dimensions in y (rows) and x (columns)
    num_rows, num_cols = terrain_array.shape
    num_patches_y = (num_rows - 1) // (patch_size - 1)
    num_patches_x = (num_cols - 1) // (patch_size - 1)

    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Calculate patch boundaries with one pixel overlap
            y_start = i * (patch_size - 1)
            x_start = j * (patch_size - 1)
            y_end = min(y_start + patch_size, num_rows)
            x_end = min(x_start + patch_size, num_cols)

            # Extract patch with overlapping boundaries
            patch = terrain_array[y_start:y_end, x_start:x_end]
            
            patch_image = Image.fromarray(patch)
            patch_filename = os.path.join(output_folder, f"patch_{i}_{j}.bmp")
            # patch_image.save(patch_filename)

            # Calculate center position
            x_center, y_center, z_center = get_patch_center_position(i, j, patch_size, bmp_dim_y, bmp_dim_x)

            patches.append((patch_filename, i, j, (x_center, y_center, z_center)))

    return patches

def assign_textures_to_patches(patches, num_clusters, patch_size, available_labels,
                               is_rigid, is_deformable):
    """
    Assign textures to patches ensuring that same texture types have contiguous boundaries.
    Example: 
    - patch_texture_map: {
        'patches/patch_0_0.bmp': 4,  # gravel
        'patches/patch_0_1.bmp': 4,  # gravel
        'patches/patch_1_0.bmp': 4,  # gravel
        'patches/patch_1_1.bmp': 8   # mud 
    }
    - terrain_labels: 129x129 array with values 4 and 8 in appropriate regions
    - cluster_centers: [(0,0), (1,1)]
    - selected_terrain_labels: [4, 8]
    """
    assert terrain_path is not None, "Terrain path must be provided."
    assert len(available_labels) > 0, "Must provide at least one available label"
    
    rigid_labels = [label for label in available_labels if label < 7]
    deformable_labels = [label for label in available_labels if label >= 7]
    
    if is_rigid and is_deformable:
        num_rigid_clusters = min(random.randint(1, num_clusters - 1), len(rigid_labels))
        num_deformable_clusters = min(num_clusters - num_rigid_clusters, len(deformable_labels))
    elif is_rigid:
        num_rigid_clusters = min(num_clusters, len(rigid_labels)) #1~7 rigid
        num_deformable_clusters = 0
    elif is_deformable:
        num_rigid_clusters = 0
        num_deformable_clusters = min(num_clusters, len(deformable_labels))
    else:
        raise ValueError("At least one of is_rigid or is_deformable must be True")

    print(f"Number of rigid clusters: {num_rigid_clusters}")
    print(f"Number of deformable clusters: {num_deformable_clusters}")
    
    # Evenly select labels for each type
    selected_rigid_labels = random.sample(rigid_labels, num_rigid_clusters)
    selected_deformable_labels = random.sample(deformable_labels, num_deformable_clusters)
    selected_terrain_labels = selected_rigid_labels + selected_deformable_labels
    print(f"Selected rigid labels: {selected_rigid_labels}")
    print(f"Selected deformable labels: {selected_deformable_labels}")
    
    # Extract patch indices (y, x coordinates)
    patch_indices = [(i, j) for _, i, j, _ in patches] # i is y-coord, j is x-coord
    patch_coords = np.array(patch_indices)
    mean_y = np.mean(patch_coords[:, 0])
    mean_x = np.mean(patch_coords[:, 1])
    std_y = np.std(patch_coords[:, 0])
    std_x = np.std(patch_coords[:, 1])
    
    cluster_centers = []
    num_clusters = num_rigid_clusters + num_deformable_clusters
    while len(cluster_centers) < num_clusters:
        y = int(np.random.normal(mean_y, std_y))
        x = int(np.random.normal(mean_x, std_x))
        
        if (y, x) in patch_indices and (y, x) not in cluster_centers:
            cluster_centers.append((y, x))
    
    # Assign labels based on closest cluster center
    label_assignments = {}
    patch_texture_map = {}
    cluster_to_label = {cluster: label for cluster, label in enumerate(selected_terrain_labels)}
    
    # Assign textures and labels to patches
    for patch_file, i, j, _ in patches:
        distances = [np.sqrt((i - center[0])**2 + (j - center[1])**2) for center in cluster_centers]
        closest_cluster = np.argmin(distances)
        closest_cluster = min(closest_cluster, len(selected_terrain_labels) - 1)
        terrain_label = cluster_to_label[closest_cluster]
        label_assignments[(i, j)] = terrain_label
        patch_texture_map[patch_file] = terrain_label
    
    # Update terrain labels
    bmp_dim_y, bmp_dim_x = terrain_array.shape
    terrain_labels = np.full((bmp_dim_y, bmp_dim_x), -1, dtype=np.int32)
    
    for (i, j), label in label_assignments.items():
        y_start = i * (patch_size - 1)
        x_start = j * (patch_size - 1)
        y_end = min(y_start + patch_size, terrain_array.shape[0])
        x_end = min(x_start + patch_size, terrain_array.shape[1])
        terrain_labels[y_start:y_end, x_start:x_end] = label
    
    return patch_texture_map, terrain_labels, cluster_centers, selected_terrain_labels

def texture_dictionary(base_path):
    """
    Create a dictionary of terrain types and their corresponding file paths.
    Each terrain type points to a randomly selected image file from its folder.
    Example: texture_dict = {'clay': 'terrain/textures/rigid/clay/clay1.jpg'}
             label_dict = {'clay': 0}   
    """
    rigid_types = ['clay', 'concrete', 'dirt', 'grass', 'gravel', 'rock', 'wood']
    deformable_types = ['mud', 'sand', 'snow']
    
    rigid_texture_dict = {}
    rigid_label_dict = {}
    deformable_texture_dict = {}
    deformable_label_dict = {}

    # Process rigid textures
    rigid_path = os.path.join(base_path, 'rigid')
    for idx, terrain in enumerate(rigid_types):
        folder_path = os.path.join(rigid_path, terrain)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                selected_texture = os.path.join(folder_path, random.choice(files))
                relative_texture_path = os.path.relpath(selected_texture, base_path)
                rigid_texture_dict[terrain] = 'terrain/textures/' + relative_texture_path
                rigid_label_dict[terrain] = idx
            else:
                print(f"No valid images found in {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")
    
    # Process deformable textures
    deformable_path = os.path.join(base_path, 'deformable')
    for idx, terrain in enumerate(deformable_types):
        folder_path = os.path.join(deformable_path, terrain)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                selected_texture = os.path.join(folder_path, random.choice(files))
                relative_texture_path = os.path.relpath(selected_texture, base_path)
                deformable_texture_dict[terrain] = 'terrain/textures/' + relative_texture_path
                deformable_label_dict[terrain] = idx + 7
            else:
                print(f"No valid images found in {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")
            
    return rigid_texture_dict, rigid_label_dict, deformable_texture_dict, deformable_label_dict

def terrain_texture_options(rigid_texture_dict, rigid_label_dict, 
                            deformable_texture_dict, deformable_label_dict):
    """
    Creates an unified dictionary of texture options for all terrain types.
    Example: 0: {
                'texture_file': 'terrain/textures/rigid/clay/clay1.jpg',
                'terrain_type': 'clay',
                'is_deformable': False
            }
    """
    label_to_rigid = {v: k for k, v in rigid_label_dict.items()}
    label_to_deformable = {v: k for k, v in deformable_label_dict.items()}
    
    texture_options = {}
    # Add rigid terrain textures
    for label, terrain_type in label_to_rigid.items():
        texture_options[label] = {
            'texture_file': rigid_texture_dict[terrain_type],
            'terrain_type': terrain_type,
            'is_deformable': False
        }
        
    # Add deformable terrain textures
    for label, terrain_type in label_to_deformable.items():
        texture_options[label] = {
            'texture_file': deformable_texture_dict[terrain_type],
            'terrain_type': terrain_type,
            'is_deformable': True
        }
    
    return texture_options

def sample_patch_properties(terrain_patches, patch_texture_map, texture_options):
    """
    Set material properties for a terrain patch based on its type.
    Example: 
    - property_dict: {
        (0, 0): {
            'is_deformable': False,
            'terrain_type': 'grass',
            'texture_file': 'terrain/textures/rigid/grass/grass2.jpg',
            'friction': 0.38,  # Sampled value
            'restitution': 0.01
        },
        (1, 1): {
            'is_deformable': True,
            'terrain_type': 'mud',
            'texture_file': 'terrain/textures/deformable/mud/mud1.jpg'
        }
    """
    property_dict = {}
    rigid_properties = RigidProperties()
    
    for patch_file, i, j, _ in terrain_patches:
        terrain_label = patch_texture_map[patch_file]
        terrain_info = texture_options[terrain_label]
        
        if terrain_info['is_deformable']:
            # For deformable terrains, set default properties
            property_dict[(i, j)] = {
                'is_deformable': True,
                'terrain_type': terrain_info['terrain_type'],
                'texture_file': terrain_info['texture_file']
            }
        else:
            # For rigid terrains, sample friction and restitution
            friction, restitution = rigid_properties.sample_properties(terrain_info['terrain_type'])
            property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': terrain_info['terrain_type'],
                'texture_file': terrain_info['texture_file'],
                'friction': friction,
                'restitution': restitution
            }
    
    return property_dict

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

def combine_rigid(m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
    """
    Attempts to combine patches into larger rectangular sections to reduce render resources
    For any remaining rigid patches that couldn't be combined, it creates individual terrain sections
    """
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
            section_pos = chrono.ChVector3d(avg_x, avg_y, 0)
            
            if not selected_patches:
                raise ValueError("No patches selected for merging.")
            
            # Check if selected patches have the same properties
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
                                               width - 1, height - 1)
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
                                             width - 1, height - 1,
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
            patch_pos = chrono.ChVector3d(*center_pos)
            
            rigid_terrain = veh.RigidTerrain(m_system)
            patch_mat = chrono.ChContactMaterialNSC()
            patch_mat.SetFriction(properties['friction'])
            patch_mat.SetRestitution(properties['restitution'])
            
            if m_isFlat:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                             patch_size - 1, patch_size - 1)
            else:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                             patch_file,
                                             patch_size - 1, patch_size - 1,
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
        section_pos = chrono.ChVector3d(chrono_center_x + 0.5, chrono_center_y - 0.5, 0)
            
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
        width = 2 * m_terrain_length - 1
        height = 2 * m_terrain_width - 1
        
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
            section_pos = chrono.ChVector3d(chrono_center_x + 0.5, chrono_center_y - 0.5, 0)
            
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
            width = 2 * m_terrain_length - 1
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
            section_pos = chrono.ChVector3d(chrono_center_x + 0.5, chrono_center_y - 1, 0)
            
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
            
            width = 2 * m_terrain_length - 1
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
    """
    Attempt to group each deformable terrain type into larger rectangular sections.
    For isolated deformable patches, convert them to the first valid rigid texture type.
    """
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
        section_pos = chrono.ChVector3d(avg_x, avg_y, 0)
        
        # Initialize terrain section
        deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
        deform_terrain.SetMeshWireframe(False)
        
        # Create and set boundary
        aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
        deform_terrain.SetBoundary(aabb)  
        
        if m_isFlat:
            deform_terrain.Initialize(width - 1, height - 1, terrain_delta)
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
                width - 1, height - 1,
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
    first_rigid_type = texture_options[first_rigid_label]['terrain_type']
    friction, restitution = RigidProperties().sample_properties(first_rigid_type)
    
    updated_property_dict = property_dict.copy()
    for patch_file, i, j, center_pos in terrain_patches:
        if (i, j) not in processed_patches and texture_options[terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]]['is_deformable']:
            updated_property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': first_rigid_type,
                'texture_file': texture_options[first_rigid_label]['texture_file'],
                'friction': friction,
                'restitution': restitution
            }
            terrain_labels[i * (patch_size - 1):(i + 1) * (patch_size - 1), 
                         j * (patch_size - 1):(j + 1) * (patch_size - 1)] = first_rigid_label

    return deformable_sections, updated_property_dict, terrain_labels

def single_terrain_patch(m_system, m_isFlat):
    if m_isFlat:
        m_terrain = veh.RigidTerrain(m_system)
        patch_mat = chrono.ChContactMaterialNSC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_pos = chrono.ChCoordsysd(chrono.ChVector3d(0, 0, -1.0), chrono.CSYSNORM.rot)
        patch = m_terrain.AddPatch(patch_mat, patch_pos, m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        patch.SetTexture(veh.GetDataFile("terrain/textures/rigid/concrete/concrete1.jpg"), 
                         m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        m_terrain.Initialize()
        
    else:
        m_terrain = veh.RigidTerrain(m_system)
        patch_mat = chrono.ChContactMaterialNSC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_pos = chrono.ChCoordsysd(chrono.ChVector3d(0, 0, -1.0), chrono.CSYSNORM.rot)
        patch = m_terrain.AddPatch(
            patch_mat, patch_pos, terrain_path, 
            m_terrain_length * 2 - 1, m_terrain_width * 2 - 1, 
            m_min_terrain_height, m_max_terrain_height
        )
        patch.SetTexture(veh.GetDataFile("terrain/textures/rigid/concrete/concrete1.jpg"), 
                         m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        m_terrain.Initialize()
        
    return m_terrain

def add_obstacles(m_system, m_goal, m_chassis_body, difficulty, pairs, m_isFlat=False):
    rock_counts = {
        'sparse': 10,    
        'medium': 20,   
        'dense': 40    
    }
    
    tree_counts = {
        'sparse': 1,    
        'medium': 2,   
        'dense': 3    
    }
    
    if difficulty not in rock_counts:
        raise ValueError(f"Invalid difficulty level. Must be one of {list(rock_counts.keys())}")
    
    rock_count = rock_counts[difficulty]
    tree_count = tree_counts[difficulty]
    
    m_assets = SimulationAssets(m_system, m_terrain_length * 1.8, m_terrain_width * 1.8, 1.0,
                                high_res_data, m_min_terrain_height, m_max_terrain_height, m_isFlat)
    
    # Add rocks
    for _ in range(rock_count):
        rock_scale = random.uniform(0.7, 1.2)
        rock = Asset(visual_shape_path="sensor/offroad/rock.obj",
                    scale=rock_scale, 
                    bounding_box=chrono.ChVector3d(4.4, 4.4, 3.8))
        m_assets.AddAsset(rock, number=1)
    
    # Add trees
    tree = Asset(visual_shape_path="sensor/offroad/tree.obj",
                scale=1.0, bounding_box=chrono.ChVector3d(1.0, 1.0, 5.0))
    m_assets.AddAsset(tree, number=tree_count)
    
    avoid_positions = []
    for start_pos, goal_pos in pairs:
        avoid_positions.append(start_pos)
        avoid_positions.append(goal_pos)
    
    obstacles_info = m_assets.RandomlyPositionAssets(m_goal, m_chassis_body, avoid_positions)
    
    return m_assets, obstacles_info

def save_terrain_labels(terrain_labels, output_path):
    """
    Save terrain labels as a single numpy array
    """
    # Save the complete label array
    np.save(output_path, terrain_labels)
    print(f"Saved terrain labels to {output_path}")

def save_config(terrain_params, pairs, terrain_type, obstacle_density, difficulty, 
                property_dict, terrain_patches, obstacles_info, m_isFlat, obstacles_flag, config_path):
    
    positions = []
    for start_pos, goal_pos in pairs:
        if isinstance(start_pos, (np.ndarray, tuple)):
            start_pos = [float(x) for x in start_pos]
        if isinstance(goal_pos, (np.ndarray, tuple)):
            goal_pos = [float(x) for x in goal_pos]
        positions.append({
            'start': start_pos,
            'goal': goal_pos
        })
    
    patch_positions = {}
    for patch_file, i, j, center_pos in terrain_patches:
        patch_positions[(i, j)] = {
            'x': float(center_pos[0]),
            'y': float(center_pos[1]),
            'z': float(center_pos[2])
        }
        
    patch_properties = []
    for (i, j), properties in property_dict.items():
        patch_info = {
            'index': [i, j],
            'center_position': patch_positions.get((i, j), {'x': 0.0, 'y': 0.0, 'z': 0.0}),
            'is_deformable': properties['is_deformable'],
            'terrain_type': properties['terrain_type'],
            'texture_file': properties['texture_file']
        }
        
        if not properties['is_deformable']:
            if 'friction' in properties:
                friction = properties['friction']
                if isinstance(friction, np.ndarray):
                    friction = float(friction.item())
                elif isinstance(friction, np.float64):
                    friction = float(friction)
                patch_info['friction'] = friction
                
            if 'restitution' in properties:
                patch_info['restitution'] = float(properties['restitution'])
        
        patch_properties.append(patch_info)
    
    obstacles_data = {
        'rocks': [
            {
                'position': {'x': float(pos['position'][0]), 
                             'y': float(pos['position'][1]), 
                             'z': float(pos['position'][2])},
                'scale': float(pos['scale'])
            } for pos in obstacles_info['rocks']
        ],
        'trees': [
            {
                'position': {'x': float(pos['position'][0]), 
                             'y': float(pos['position'][1]), 
                             'z': float(pos['position'][2])}
            } for pos in obstacles_info['trees']
        ]
    }
       
    config_data = {
        'terrain': {
            'length': float(terrain_params['length']),
            'width': float(terrain_params['width']),
            'min_height': float(terrain_params['min_height']),
            'max_height': float(terrain_params['max_height']),
            'difficulty': difficulty,
            'is_flat': m_isFlat
        },
        'positions': positions,
        'terrain_type': terrain_type,
        'obstacles_flag': obstacles_flag,
        'obstacle_density': obstacle_density,
        'textures': patch_properties,
        'obstacles': obstacles_data
    }

    with open(config_path, 'w') as file:
        yaml.dump(config_data, file, default_flow_style=None)
    
    print(f"Config saved to {config_path}")

def gen_obs_map(terrain_array, obstacles_info, m_terrain_length, m_terrain_width):
    """
    Generate modified terrain bitmap with obstacles marked
    0 - free space, 255 - obstacle
    """
    bmp_dim_y, bmp_dim_x = terrain_array.shape
    obs_terrain = terrain_array.copy()
    
    # Process rocks
    for rock in obstacles_info['rocks']:
        # Get rock position and scale
        rock_pos = rock['position'] 
        rock_scale = rock['scale']
        
        # Create ChVector3d for position
        obstacle_pos = chrono.ChVector3d(rock_pos[0], rock_pos[1], rock_pos[2])
        
        # Transform obstacle position to bitmap coordinates
        obstacle_bmp = transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
        obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, bmp_dim_x - 1)))
        obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, bmp_dim_y - 1)))
        
        # Rock bounding box is 4.4 x 4.4 x 3.8
        box_width = 4.4 * rock_scale
        box_length = 4.4 * rock_scale
            
        # Create a mask for the obstacle
        width_pixels = int(box_width * bmp_dim_x / (2 * m_terrain_length))
        length_pixels = int(box_length * bmp_dim_x / (2 * m_terrain_width))
        
        # Calculate bounds for the obstacle footprint
        x_min = max(0, obs_x - width_pixels // 2)
        x_max = min(bmp_dim_x, obs_x + width_pixels // 2 + 1)
        y_min = max(0, obs_y - length_pixels // 2)
        y_max = min(bmp_dim_y, obs_y + length_pixels // 2 + 1)
        
        obs_terrain[y_min:y_max, x_min:x_max] = 255
    
    # Process trees
    for tree in obstacles_info['trees']:
        # Get tree position
        tree_pos = tree['position'] 
        
        # Create ChVector3d for position
        obstacle_pos = chrono.ChVector3d(tree_pos[0], tree_pos[1], tree_pos[2])
        
        # Transform obstacle position to bitmap coordinates
        obstacle_bmp = transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
        obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, bmp_dim_x - 1)))
        obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, bmp_dim_y - 1)))
        
        # Tree bounding box is 1.0 x 1.0 x 5.0
        box_width = 1.0
        box_length = 1.0
            
        # Create a mask for the obstacle
        width_pixels = int(box_width * bmp_dim_x / (2 * m_terrain_length))
        length_pixels = int(box_length * bmp_dim_x / (2 * m_terrain_width))
        
        # Calculate bounds for the obstacle footprint
        x_min = max(0, obs_x - width_pixels // 2)
        x_max = min(bmp_dim_x, obs_x + width_pixels // 2 + 1)
        y_min = max(0, obs_y - length_pixels // 2)
        y_max = min(bmp_dim_y, obs_y + length_pixels // 2 + 1)
        
        obs_terrain[y_min:y_max, x_min:x_max] = 255
    
    # Save modified terrain
    obs_terrain_image = Image.fromarray(obs_terrain.astype(np.uint8), mode='L')
    obs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "./data/BenchMaps/sampled_maps/Configs/Custom", f"obs{world_id}_{difficulty}.bmp")
    os.makedirs(os.path.dirname(obs_path), exist_ok=True)
    obs_terrain_image.save(obs_path)
    
    return obs_path

def run_simulation(pairs, render=False, use_gui=False, m_isFlat = False, is_rigid=False, is_deformable=False, obstacles_flag=False):
    if not (is_rigid or is_deformable):
        raise ValueError("At least one terrain type must be enabled")
    
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    num_procs = multiprocessing.cpu_count()
    num_threads_chrono = min(8, num_procs)
    num_threads_collision = min(8, num_procs)
    num_threads_eigen = 1
    m_system.SetNumThreads(num_threads_chrono, num_threads_collision, num_threads_eigen)
    
    # Visualization frequencies
    m_vis_freq = 100.0  # Hz
    m_vis_dur = 1.0 / m_vis_freq
    last_vis_time = 0.0
    
    # Setup terrain parameters
    patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  "./data/BenchMaps/sampled_maps/patches")
    patch_size = 9
    terrain_patches = divide_terrain_image(patches_folder, patch_size)

    available_labels = []
    if is_rigid:
        available_labels.extend(list(range(7)))  # Rigid labels 0-6
    if is_deformable:
        available_labels.extend(list(range(7, 10)))  # Deformable labels 7-9
    
    if is_rigid and is_deformable:
        min_clusters = 2
        num_clusters = random.randint(min_clusters, len(available_labels))
    elif is_rigid or is_deformable:
        min_clusters = 1
        num_clusters = random.randint(min_clusters, len(available_labels))
    
    patch_texture_map, terrain_labels, \
    cluster_centers, selected_terrain_labels = assign_textures_to_patches(
        patches=terrain_patches,
        num_clusters=num_clusters,
        patch_size=patch_size,
        available_labels=available_labels,
        is_rigid=is_rigid,
        is_deformable=is_deformable
    )
    
    # Patch and obstacles creation
    property_dict = sample_patch_properties(terrain_patches, patch_texture_map, texture_options)
    obstacles_info = {
        'rocks': [],
        'trees': []
    }
    
    rigid_terrains = []
    deform_terrains = []
    if is_rigid and not is_deformable:
        rigid_terrains, property_dict, terrain_labels = combine_rigid(
            m_system, terrain_patches, terrain_labels, property_dict, 
            texture_options, patch_size, m_isFlat
        )
            
    elif is_deformable and not is_rigid:
        deform_terrains = combine_deformation(m_system, terrain_patches, property_dict, texture_options, m_isFlat)
        
    elif is_deformable and is_rigid:
        deform_terrains, property_dict, terrain_labels = mixed_terrain(
            m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat
        )
        rigid_terrains, property_dict, terrain_labels = combine_rigid(
            m_system, terrain_patches, terrain_labels, property_dict, 
            texture_options, patch_size, m_isFlat
        )
    
    # # Make sure single terrain patch is same as small patches
    # background_terrain = single_terrain_patch(m_system, m_isFlat)
    
    # Generate high resolution terrain labels
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
    
    # If rendering, randomly select (start, goal) pair
    start_pos, goal_pos = pairs[pair_id]
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
    
    if is_deformable:
        for deform_terrain in deform_terrains:
            for axle in m_vehicle.GetVehicle().GetAxles():
                deform_terrain.AddMovingPatch(axle.m_wheels[0].GetSpindle(), chrono.VNULL, chrono.ChVector3d(1.0, 0.6, 1.0))
                deform_terrain.AddMovingPatch(axle.m_wheels[1].GetSpindle(), chrono.VNULL, chrono.ChVector3d(1.0, 0.6, 1.0))
            deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    
    if obstacles_flag:
        m_assets, obstacles_info = add_obstacles(m_system, m_goal, m_chassis_body, difficulty=obstacle_density, pairs=pairs, m_isFlat=m_isFlat)
        # Save obstacles map
        obs_path = gen_obs_map(terrain_array, obstacles_info, m_terrain_length, m_terrain_width)
       
    # Save terrain labels
    labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                               "./data/BenchMaps/sampled_maps/Configs/Custom")
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)
    labels_file = os.path.join(labels_path, f"labels{world_id}_{difficulty}.npy")
    save_terrain_labels(terrain_labels, labels_file)
    
    # Save terrain configuration
    terrain_params = {
        'length': m_terrain_length,
        'width': m_terrain_width,
        'min_height': m_min_terrain_height,
        'max_height': m_max_terrain_height,
    }
    
    config_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "./data/BenchMaps/sampled_maps/Configs/Custom",
        f"config{world_id}_{difficulty}.yaml"
    )
    save_config(
        terrain_params, pairs, selected_option, obstacle_density, difficulty,
        property_dict, terrain_patches, obstacles_info, m_isFlat, obstacles_flag, config_path
    )
    
    # Visualization
    if render:
        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle('vws in the wild')
        vis.SetWindowSize(2560, 1440)
        trackPoint = chrono.ChVector3d(0.0, 0.0, 1.75)
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
    m_speedController.SetGains(1, 0, 0)
    # Initialize the custom PID controller for steering
    m_steeringController = PIDController(kp=1.0, ki=0.0, kd=0.0)
    
    # Continuous speed
    speed = 4.0 if not use_gui else 0.0 
    start_time = m_system.GetChTime()
    
    roll_angles = []
    pitch_angles = []
    
    # # Cropped labels visualization
    # plt.ion()
    # fig, ax = plt.subplots(figsize=(4,4))
    # im = ax.imshow(np.zeros((8,8)), cmap='tab20', interpolation='nearest')
    # ax.set_title('Current Label')
    # ax.axis('off')
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    
    # When generating config, comment following code to return
    while True:
        if render and not vis.Run():
            break
        
        time = m_system.GetChTime()
        # Render at low frequency
        if render and (last_vis_time==0 or (time - last_vis_time) > m_vis_dur):
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            last_vis_time = time
            
        m_vehicle_pos = m_vehicle.GetChassisBody().GetPos() #Global coordinate
        m_vector_to_goal = m_goal - m_vehicle_pos
        
        if use_gui:
            m_driver_inputs = m_driver.GetInputs()
        else:
            goal_heading = np.arctan2(m_vector_to_goal.y, m_vector_to_goal.x)
                    
            euler_angles = m_vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ() #Global coordinate
            roll = euler_angles.x
            pitch = euler_angles.y
            vehicle_heading = euler_angles.z
            heading_error = (goal_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            
            roll_angles.append(np.degrees(abs(roll)))
            pitch_angles.append(np.degrees(abs(pitch)))

            #PID controller for steering
            steering = -m_steeringController.compute(heading_error, m_step_size)
            m_driver_inputs.m_steering = np.clip(steering, m_driver_inputs.m_steering - 0.05, 
                                                 m_driver_inputs.m_steering + 0.05)
            
            # # PID + ElevationMap controller for steering
            # region_size = 128
            # under_vehicle, front_regions = get_cropped_map(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 
            #                                                region_size, 5)
        
            # # # Visualize cropped height map
            # # normalized_under_vehicle = ((under_vehicle - m_min_terrain_height) / (m_max_terrain_height - m_min_terrain_height) * 255).astype(np.uint8)
            # # cv2.imshow("Rotated Cropp Height Map", normalized_under_vehicle)
            # # cv2.waitKey(1)
            
            # relative_x, relative_y = find_similar_region(under_vehicle, front_regions, region_size)
            # target_heading = np.arctan2(relative_y, relative_x)
            # target_heading_error = (target_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            
            # target_heading_weight = 0.3
            # goal_heading_weight = 0.7
            # combined_error = target_heading_weight * target_heading_error + goal_heading_weight * heading_error
            # steering = -m_steeringController.compute(combined_error, m_step_size)
            # m_driver_inputs.m_steering = np.clip(steering, m_driver_inputs.m_steering - 0.05,
            #                                      m_driver_inputs.m_steering + 0.05)
            
            # Desired throttle/braking value
            out_throttle = m_speedController.Advance(m_vehicle.GetRefFrame(), speed, time, m_step_size)
            out_throttle = np.clip(out_throttle, -1, 1)
            if out_throttle > 0:
                m_driver_inputs.m_braking = 0
                m_driver_inputs.m_throttle = out_throttle
            else:
                m_driver_inputs.m_braking = -out_throttle
                m_driver_inputs.m_throttle = 0
        
        # # Semantic Labels
        # region_size = 64
        # current_label = get_current_label(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 
        #                                   region_size, high_res_terrain_labels)
        # print("Current label:\n", current_label)
        
        # # Cropped labels visualization
        # im.set_data(current_label)
        # im.set_clim(vmin=current_label.min(), vmax=current_label.max())
        # fig.canvas.draw()
        # fig.canvas.flush_events()
            
        if m_vector_to_goal.Length() < 8:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            avg_roll = np.mean(roll_angles) if roll_angles else 0 
            avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
            return time - start_time, True, avg_roll, avg_pitch
        
        if m_system.GetChTime() > m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', m_initLoc)
            dist = m_vector_to_goal.Length()
            print('Final position of art: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            avg_roll = np.mean(roll_angles) if roll_angles else 0 
            avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
            return time - start_time, False, avg_roll, avg_pitch
        
        if is_rigid:
            # print("Rigid terrain", len(rigid_terrains))
            for rigid_terrain in rigid_terrains:
                rigid_terrain.Synchronize(time)
                m_vehicle.Synchronize(time, m_driver_inputs, rigid_terrain)
                rigid_terrain.Advance(m_step_size)
            
        if is_deformable:
            # print("Deform terrain", len(deform_terrains))
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
    
    return None, False, 0, 0  # Return None if goal not reached

if __name__ == '__main__':
    # Terrain parameters
    SetChronoDataDirectories()
    CHRONO_DATA_DIR = chrono.GetChronoDataPath()
    base_texture_path = os.path.join(CHRONO_DATA_DIR, "vehicle/terrain/textures/")
    rigid_texture_dict, rigid_label_dict, \
    deformable_texture_dict, deformable_label_dict = texture_dictionary(base_texture_path)
    texture_options = terrain_texture_options(
        rigid_texture_dict, 
        rigid_label_dict,
        deformable_texture_dict, 
        deformable_label_dict
    )
    
    print("Rigid texture labels:")
    print(rigid_label_dict)
    print("Deformable texture labels:")
    print(deformable_label_dict)
    
    #====================Base Params===========================
    # Geometry difficulty
    # X-direction: front, Y-direction: left, Z-direction: up
    m_terrain_length = 64.5  # half size in X direction
    m_terrain_width = 64.5  # half size in Y direction 
    terrain_delta = 0.1 # mesh resolution for SCM terrain
    
    # Simulation step sizes
    m_max_time = 5
    m_step_size = 5e-3 # simulation update every num seconds
    
    # Start and goal pairs
    num_pairs = 10
    circle_radius = 60
    circle_center = (0, 0, 1.0)
    #====================Base Params===========================
    
    # Generate configurations for worlds 1-100
    for world_id in range(1, 101):
        if not 1 <= world_id <= 100:
            raise ValueError("Check world id (1-100)!")
        
        print(f"\nProcessing World {world_id}")
        print("-" * 50)
        
        # Load terrain bitmap
        terrain_file = f"{world_id}.bmp"
        terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "./data/BenchMaps/sampled_maps/Worlds", terrain_file)
        terrain_image = Image.open(terrain_path)
        terrain_array = np.array(terrain_image)
        bmp_dim_y, bmp_dim_x = terrain_array.shape 
        if (bmp_dim_y, bmp_dim_x) != (129, 129):
            raise ValueError("Check terrain file and dimensions")
        
        # Load high resolution bitmap
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
            raise ValueError("Check high resolution height map dimensions")
        
        # Extract difficulty from bitmap name
        actual_filename = os.path.basename(actual_file_path)
        match = re.search(r'height\d+_(\w+)\.npy', actual_filename)
        difficulty = match.group(1)
        terrain_difficulty = GeometryDiff()
        m_min_terrain_height, m_max_terrain_height = terrain_difficulty.get_height_range(difficulty)
        
        # Generate start and goal pairs
        pairs = generate_circle_pairs(circle_center, circle_radius, num_pairs, m_terrain_length, m_terrain_width)
        pair_id = random.choice(range(num_pairs))
        if not pairs:
            raise ValueError("No valid pairs within terrain boundaries. Check terrain size and circle radius.")
        
        # Randomly select obstacles level by 1/3 probability
        obstacle_densities = ['sparse', 'medium', 'dense']
        obstacle_density = random.choice(obstacle_densities)
        
        # Select terrain type
        terrain_options = ['rigid', 'deformable', 'mixed']
        if world_id <= 60:
            selected_option = 'rigid'
        elif world_id <= 90:
            selected_option = 'deformable'
        else:
            selected_option = 'mixed'
        
        if selected_option == 'rigid':
            is_rigid = True
            is_deformable = False
        elif selected_option == 'deformable':
            is_rigid = False
            is_deformable = True
        else:
            is_rigid = True
            is_deformable = True
            
        print(f"Configuration for World {world_id}:")
        print(f"Terrain Type: {selected_option}")
        print(f"Difficulty: {difficulty}")
        print(f"Obstacle Density: {obstacle_density}")
        
        time_to_goal, success, avg_roll, avg_pitch = run_simulation(pairs, render=False, use_gui=False, m_isFlat=False,
                                                                    is_rigid=is_rigid, is_deformable=is_deformable, obstacles_flag=True)
        
    print("Finish generating configurations for worlds!") 
    