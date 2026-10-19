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

def get_interpolated_height(terrain_array, px_float, py_float, hMin, hMax):
    """
    Get interpolated height value using bilinear interpolation.
    """
    nv_y, nv_x = terrain_array.shape[:2]
    px_float_clamped = np.clip(px_float, 0, nv_x - 1 - 1e-9)
    py_float_clamped = np.clip(py_float, 0, nv_y - 1 - 1e-9)
    
    px0 = int(np.floor(px_float_clamped))
    py0 = int(np.floor(py_float_clamped))
    px1 = min(px0 + 1, nv_x - 1)
    py1 = min(py0 + 1, nv_y - 1)
    
    # Calculate the fractional parts
    tx = px_float_clamped - px0
    ty = py_float_clamped - py0
    
    h00 = terrain_array[py0, px0] / 255.0
    h10 = terrain_array[py0, px1] / 255.0 # Top-right
    h01 = terrain_array[py1, px0] / 255.0 # Bottom-left
    h11 = terrain_array[py1, px1] / 255.0 # Bottom-right
    
    # Interpolate horizontally (along x) for top and bottom edges
    h_top = h00 * (1 - tx) + h10 * tx
    h_bottom = h01 * (1 - tx) + h11 * tx

    # Interpolate vertically (along y) between the intermediate values
    h_ratio = h_top * (1 - ty) + h_bottom * ty

    # Scale the interpolated ratio to the physical height range
    final_height = hMin + h_ratio * (hMax - hMin)

    return final_height

def initialize_vw_pos(m_vehicle, start_pos, goal_pos, m_isFlat):
    if m_isFlat:
        start_height = 0
    else:
        pos_bmp = transform_to_bmp([start_pos])[0]    
        start_height = get_interpolated_height(terrain_array, pos_bmp[0], pos_bmp[1], m_min_terrain_height, m_max_terrain_height)

    start_pos = (start_pos[0], start_pos[1], start_height + start_pos[2])
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    start_yaw = np.arctan2(dy, dx)
    m_initLoc = chrono.ChVector3d(*start_pos)
    m_initRot = chrono.QuatFromAngleZ(start_yaw)
    m_vehicle.SetInitPosition(chrono.ChCoordsysd(m_initLoc, m_initRot))
    return m_initLoc, m_initRot, start_yaw

def set_goal(m_system, goal_pos, m_isFlat):
    if m_isFlat:
        goal_height = 0
    else:
        pos_bmp = transform_to_bmp([goal_pos])[0]
        goal_height = get_interpolated_height(terrain_array, pos_bmp[0], pos_bmp[1], m_min_terrain_height, m_max_terrain_height)
        
    goal_pos = (goal_pos[0], goal_pos[1], goal_height + goal_pos[2])
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

def single_terrain_patch(m_system, m_isFlat):
    if m_isFlat:
        m_terrain = veh.RigidTerrain(m_system)
        patch_mat = chrono.ChContactMaterialNSC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_pos = chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.0), chrono.CSYSNORM.rot)
        patch = m_terrain.AddPatch(patch_mat, patch_pos, m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        patch.SetTexture(veh.GetDataFile("terrain/textures/rigid/concrete/concrete1.jpg"), 
                         m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        m_terrain.Initialize()
        
    else:
        m_terrain = veh.RigidTerrain(m_system)
        patch_mat = chrono.ChContactMaterialNSC()
        patch_mat.SetFriction(0.9)
        patch_mat.SetRestitution(0.01)
        patch_pos = chrono.ChCoordsysd(chrono.ChVector3d(0, 0, 0.0), chrono.CSYSNORM.rot)
        patch = m_terrain.AddPatch(
            patch_mat, patch_pos, terrain_path, 
            m_terrain_length * 2 - 1, m_terrain_width * 2 - 1, 
            m_min_terrain_height, m_max_terrain_height
        )
        patch.SetTexture(veh.GetDataFile("terrain/textures/rigid/concrete/concrete1.jpg"), 
                         m_terrain_length * 2 - 1, m_terrain_width * 2 - 1)
        m_terrain.Initialize()
        
    return m_terrain

def dump_heightmap(m_system, m_terrain, npy_file, samples=1291):
    """
    Query mesh height and save as .npy following BMP coordinate.
    """
    #Initialize the system and advance by a tiny step
    m_system.DoStepDynamics(m_step_size)
    m_terrain.Synchronize(m_step_size)
    m_terrain.Advance(m_step_size)
    
    xs = np.linspace(-m_terrain_length + 0.5, m_terrain_length - 0.5, samples)
    ys = np.linspace(-m_terrain_width + 0.5, m_terrain_width - 0.5, samples)
    xs = np.round(xs, 6)
    ys = np.round(ys, 6)

    height_map = np.empty((samples, samples), dtype=np.float64)
    for iy, y in enumerate(ys):
        for ix, x in enumerate(xs):
            height_map[iy, ix] = m_terrain.GetHeight(chrono.ChVector3d(x, y, m_max_terrain_height))
            if ix % 100 == 0 and iy % 100 == 0:
                print(f"height at ({x}, {y}) = {height_map[iy, ix]}")
    
    np.save(npy_file, height_map)
    print(f"saved height map → {npy_file}")

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
    background_terrain = single_terrain_patch(m_system, m_isFlat)
    heightmap_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                               "./data/BenchMaps/sampled_maps/Configs/Custom")
    # Store prebuilt heightmap
    dump_heightmap(
        m_system,
        background_terrain,
        npy_file=os.path.join(heightmap_path, f"height{world_id}_{difficulty}.npy"),
        samples=1291)  
    
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
    
    # When generate config, comment following code to return
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
            
            # Desired throttle/braking value
            out_throttle = m_speedController.Advance(m_vehicle.GetRefFrame(), speed, time, m_step_size)
            out_throttle = np.clip(out_throttle, -1, 1)
            if out_throttle > 0:
                m_driver_inputs.m_braking = 0
                m_driver_inputs.m_throttle = out_throttle
            else:
                m_driver_inputs.m_braking = -out_throttle
                m_driver_inputs.m_throttle = 0
            
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
        
        
        background_terrain.Synchronize(time)
        m_vehicle.Synchronize(time, m_driver_inputs, background_terrain)
        background_terrain.Advance(m_step_size)
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
    
    #====================Base Params===========================
    # Geometry difficulty
    # X-direction: front, Y-direction: left, Z-direction: up
    m_terrain_length = 65  # half size in X direction
    m_terrain_width = 65  # half size in Y direction 
    terrain_delta = 0.1 # mesh resolution for SCM terrain
    
    # Simulation step sizes
    m_max_time = 5
    m_step_size = 5e-3 # simulation update every num seconds
    
    # Start and goal pairs
    num_pairs = 10
    circle_radius = 60
    circle_center = (0, 0, 2.0)
    #====================Base Params===========================
    
    # Generate heighmap for worlds 1-100
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
        
        # Evenly select difficulty
        difficulty_levels = ['low', 'mid', 'high']
        difficulty = random.choice(difficulty_levels)
        terrain_difficulty = GeometryDiff()
        m_min_terrain_height, m_max_terrain_height = terrain_difficulty.get_height_range(difficulty)
        
        # Generate start and goal pairs
        pairs = generate_circle_pairs(circle_center, circle_radius, num_pairs, m_terrain_length, m_terrain_width)
        pair_id = random.choice(range(num_pairs))
        if not pairs:
            raise ValueError("No valid pairs within terrain boundaries. Check terrain size and circle radius.")
            
        print(f"Configuration for World {world_id}:")
        print(f"Difficulty: {difficulty}")
        
        time_to_goal, success, avg_roll, avg_pitch = run_simulation(pairs, render=False, use_gui=False, m_isFlat=False,
                                                                    is_rigid=True, is_deformable=False, obstacles_flag=False)
    
    print("Finish storing all height maps!")   
    
        