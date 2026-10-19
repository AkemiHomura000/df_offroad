import numpy as np
import heapq
from PIL import Image
import os
import pickle
import time
import sys 

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr

import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from collections import defaultdict
from scipy.ndimage import binary_dilation
from scipy.special import comb

from .models import TAL, ElevMapEncDec, PatchDecoder
from .utilities import utils
from .Grid import MapProcessor
from .traversabilityEstimator import TravEstimator
from .KrModel import KrModel

class AStarPlanner:
    def __init__(self, grid_map, start, goal, resolution=1.0):
        self.grid_map = grid_map #A 2D array where 0 represents free space
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.resolution = resolution
        self.open_list = [] #Nodes to be evaluated
        self.closed_list = set() #Nodes already evaluated
        self.parent = {}
        self.g_costs = {}
        
    def heuristic(self, node):
        return ((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)**0.5

    def get_neighbors(self, node):
        # Get valid neighboring grid cells
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
        # A* path planning algorithm
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
        # Reconstruct path
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(self.start)
        return path[::-1]

MAX_VEL = 0.6
MIN_VEL = 0.45
wm_vct = False

class MPPI6Planner:
    def __init__(self, terrain_manager):
        #Robot Limits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.max_del = 1
        self.min_del = -1
        self.robot_length = 3.4
        self.min_speed = MIN_VEL
        self.max_speed = MAX_VEL
        self.min_steer_angle = -1.0
        self.max_steer_angle = 1.0
        self.terrain_manager = terrain_manager
        self.look_ahead_distance = 10.0 * self.terrain_manager.scale_factor
        self.inflation_radius = int(4.0 * self.terrain_manager.scale_factor)
        self.speed_controller = None 
        
        #Set module-level references
        main_module = sys.modules.get('__main__', None)
        if main_module:
            if not hasattr(main_module, 'TAL'):
                main_module.TAL = TAL
            if not hasattr(main_module, 'ElevMapEncDec'):
                main_module.ElevMapEncDec = ElevMapEncDec
            if not hasattr(main_module, 'PatchDecoder'):
                main_module.PatchDecoder = PatchDecoder
        
        #External class definations 
        self.mp = MapProcessor()                                                                    # Only for crawler
        self.util = utils() 
        
        #Utility class
        self.goal_tensor = None                                                                      # Goal tensor
        self.lasttime = None                                                                         # Last time
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "TAL_13_14_14.torch") # Default Model path                                           
        enc_dec_path = "rock_LN_TH_03-07-20-08.torch"                                               
        
        elev_map_encoder = ElevMapEncDec()
        elev_map_encoder = elev_map_encoder.cuda()
        elev_map_decoder = PatchDecoder(elev_map_encoder.map_encode)
        elev_map_decoder = elev_map_decoder.cuda()

        self.model = TAL(elev_map_encoder.map_encode, elev_map_decoder)
        self.model = self.model.cuda()
        
        state_dict = torch.load(model_path, weights_only=False).state_dict()
        self.model.load_state_dict(state_dict)                                                      # Motion model
        self.model.eval()                                                                           # Model ideally runs faster in eval mode
        self.dtype = torch.float32                                                                  # Data type
        # print("Loading:", model_path)                                                               
        # print("Model:\n",self.model)
        # print("Torch Datatype:", self.dtype)

        #------MPPI variables and constants----
        #Parameters
        self.T = 20                      # Length of rollout horizon
        self.K = 600                     # Number of sample rollouts
        self.dt = 1
        self._lambda = 0.1               # Temperature
        self.sigma = torch.Tensor([0.2, 0.5]).type(torch.float32).expand(self.T, self.K, 2).to(self.device)
        self.inv_sigma = (1 / self.sigma[0, 0, :]).to(self.device)  # (2, )
        
        stats_pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stats_rock.pickle')
        with open(stats_pickle_path, 'rb') as f:
            scale = pickle.load(f)
        
        self.scale_state = scale['pose_dot']
        self.offset_scale = scale['map_offset']

        self.robot_pose = None                                                                    # Robot pose
        self.poses = torch.Tensor(self.K, self.T, 6).type(self.dtype)
        self.fpose = torch.Tensor(self.K, 6).type(self.dtype)
        self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype)                                   # (K,6)
        self.last_pose = None
        self.at_goal = True
        self.curr_pose = None
        self.pose_dot = torch.zeros(6).type(self.dtype)
        self.last_t = 0
        self.map_origin = torch.Tensor([64.5, 64.5]).type(self.dtype)
        self.map_resolution = 1
        self.current_trajectories = None
        self.obstacle_map = None

        #Cost variables
        self.running_cost = torch.zeros(self.K).type(self.dtype)
        self.pose_cost = torch.Tensor(self.K).type(self.dtype)
        self.bounds_check = torch.Tensor(self.K).type(self.dtype)
        self.height_check = torch.Tensor(self.K).type(self.dtype)
        self.ctrl_cost = torch.Tensor(self.K, 2).type(self.dtype)
        self.ctrl_change = torch.Tensor(self.T, 2).type(self.dtype)
        self.euclidian_distance = torch.Tensor(self.K).type(self.dtype)
        self.dist_to_goal = torch.Tensor(self.K, 6).type(self.dtype)
        self.dtg_w_init_pose = torch.Tensor(self.K, 6).type(self.dtype)

        self.map_embedding = None
        self.recent_controls = np.zeros((3,2))
        self.control_i = 0
        self.msgid = 0
        self.speed = 0
        self.steering_angle = 0
        self.prev_ctrl = None
        self.ctrl = torch.zeros((self.T, 2))  # Initial speed = 5.0 m/s
        self.rate_ctrl = 0
        self.cont_ctrl = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_idx = [1,    3 ,     5,      7,      8,      9,      10,    11,        12,         13]
        # Weights for the 8 traversability maps
        self.weights = torch.tensor([0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.3, 0.3], dtype=self.dtype).cuda().unsqueeze(-1).unsqueeze(-1)
        # Initialize the traversability model
        self.model.to(self.device)
        self.traversability_model = TravEstimator(output_dim=(320,260)).to(self.device)
        trav_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'best_traversability_model.pth')
        self.traversability_model.load_state_dict(torch.load(trav_model_path, map_location=self.device))
        self.traversability_model.eval()  # Set to evaluation mode
        self.traversability_model.requires_grad_(False)  # Disable gradient computation
        self.traversability_mask = None
        self.trav_map_combined = None
        self.traversability_model(torch.zeros(2, 1, 160, 130).to(self.device))

        #6DOF model Initialization
        kr_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Kr_model_50.pth')
        checkpoint = torch.load(kr_model_path)
        self.kr_model = KrModel(5)
        self.kr_model.load_state_dict(checkpoint, strict=True)
        self.kr_model.eval()
        self.kr_model.to(self.device)

        self.times_mppi_called = 0
        self.images_for_poses = None

        stats_kr_bench = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stats_kr_bench.pkl')
        with open(stats_kr_bench, 'rb') as f:
            self.stats = pickle.load(f)
        
        self.stats['cmd_vel_mean'] = torch.tensor(self.stats['cmd_vel_mean']).cuda().float()
        self.stats['cmd_vel_std'] = torch.tensor(self.stats['cmd_vel_std']).cuda().float()
        self.stats['velocity_mean'] = torch.tensor(self.stats['velocity_mean']).cuda().float()
        self.stats['velocity_std'] = torch.tensor(self.stats['velocity_std']).cuda().float()
        self.stats['res_roll_pitch_mean'] = torch.tensor(self.stats['res_roll_pitch_mean']).cuda().float()
        self.stats['res_roll_pitch_std'] = torch.tensor(self.stats['res_roll_pitch_std']).cuda().float()
        self.stats['roll_pitch_yaw_mean'] = torch.tensor(self.stats['roll_pitch_yaw_mean']).cuda().float()
        self.stats['roll_pitch_yaw_std'] = torch.tensor(self.stats['roll_pitch_yaw_std']).cuda().float()
        
    def gridMap_callback(self, vehicle, vehicle_pos, obstacle_array):
        """
        Callback to process the elevation map and update the map embedding.
        :param bmp_elevation_map: Ground truth elevation map as a [129, 129] BMP array
        """
        if self.robot_pose is not None:

            # Crop [29, 29] region around the vehicle
            crop_size = 29
            cropped_map, _ = self.terrain_manager.get_cropped_map(vehicle, (vehicle_pos.x, vehicle_pos.y, vehicle_pos.z), crop_size, 5)

            input_size = (360, 360)
            resized_map = cv2.resize(cropped_map, input_size, interpolation=cv2.INTER_LANCZOS4)

            # Dynamically rescale elevation values
            map_min = obstacle_array.min()
            map_max = obstacle_array.max()
            resized_map = (resized_map - map_min) / (map_max - map_min) * 1.6
            
            # Convert to tensor
            map_d = torch.tensor(resized_map, dtype=self.dtype).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

            with torch.no_grad():
                # Generate traversability maps
                self.map_embedding = self.model.process_map(map_d).repeat(self.K, 1, 1, 1).cuda()
        
        else:
            print("Warning: robot_pose is not set. Cannot process elevation map.")

    def set_obstacle_map(self, obstacle_array):
        """
        Set and process the obstacle map from the main function.
        """
        self.obstacle_map = torch.tensor(obstacle_array, dtype=self.dtype).to(self.device)
    
    def goal_cb(self, local_goal):
        """
        Callback to set the goal directly using Chrono's goal representation.
        """
        # Convert Chrono goal to PyTorch tensor
        goal_tensor_new = torch.Tensor([
            local_goal[0],  # x
            local_goal[1],  # y
            1.6,  # z
            0,            # roll (default, not provided by Chrono goal)
            0,            # pitch (default, not provided by Chrono goal)
            0             # yaw (default, not provided by Chrono goal)
        ]).type(self.dtype)

        # Set the goal tensor and update status
        self.goal_tensor = goal_tensor_new
        self.at_goal = False
    
    def cost(self, pose, goal, ctrl, noise, roll, pitch, t, initial_pose):
        device = self.device
        pose = pose.to(device)
        goal = goal.to(device)
        ctrl = ctrl.to(device)
        noise = noise.to(device)
        roll = roll.to(device)
        pitch = pitch.to(device)
        initial_pose = initial_pose.to(device)
        self.fpose = self.fpose.to(device)
        self.dist_to_goal = self.dist_to_goal.to(device)
        self.dtg_w_init_pose = self.dtg_w_init_pose.to(device)
        self.euclidian_distance = self.euclidian_distance.to(device)
        self.ctrl_cost = self.ctrl_cost.to(device)
        self.running_cost = self.running_cost.to(device)
        self.inv_sigma = self.inv_sigma.to(device)
        
        self.fpose.copy_(pose)
        self.dist_to_goal.copy_(self.fpose).sub_(goal)
        self.dtg_w_init_pose.copy_(initial_pose).sub_(goal) 
        self.dist_to_goal[:,3] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,3])
        self.dist_to_goal[:,4] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,4])
        self.dist_to_goal[:,5] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,5])
        
        xy_to_goal = self.dist_to_goal[:,:2]
        self.euclidian_distance = torch.norm(xy_to_goal, p=2, dim=1)
        euclidian_distance_squared = self.euclidian_distance.pow(2)
        theta_distance_to_goal = self.dist_to_goal[:,5] % (2*np.pi)

        xy_to_goal_init = self.dtg_w_init_pose[:,:2]
        euclidian_distance_squared_init = torch.norm(xy_to_goal_init, p=2, dim=1).pow(2)

        # Calculate control cost - all on the same device now
        self.ctrl_cost.copy_(ctrl).mul_(self._lambda).mul_(self.inv_sigma).mul_(noise)
        running_cost_temp = self.ctrl_cost.abs_().sum(dim=1)
        self.running_cost.copy_(running_cost_temp)
        
        # Add different cost components - all on GPU
        self.running_cost.add_(euclidian_distance_squared)
        self.running_cost.add_((theta_distance_to_goal * 30)/euclidian_distance_squared_init[0])
        self.running_cost.add_(roll*1.0)
        self.running_cost.add_(pitch*0.5)

    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone()

        # shift all controls forward by 1, with last control replicated
        self.ctrl = torch.roll(self.ctrl, shifts=-1, dims=0)
        return run_ctrl

    def mppi(self, init_pose, init_velocities, init_angles):
        t0 = time.time()
        self.times_mppi_called += 1
        dt = self.dt
        device = self.device  # Use the class's device attribute

        # Ensure all tensors are on the correct device
        self.running_cost = self.running_cost.to(device)
        self.running_cost.zero_()
        
        # Move input tensors to device
        init_pose = init_pose.to(device)
        init_velocities = init_velocities.to(device)
        init_angles = init_angles.to(device)
        
        # Create tensors on the correct device
        cur_pose = init_pose.repeat(self.K, 1).to(device)
        init_pose_rep = cur_pose.clone() 
        pure_pose = cur_pose.clone()

        ctrl_temp = torch.zeros(self.K, 2, device=device)
        nn_velocities = init_velocities.repeat(self.K, 1, 1).to(device).type(self.dtype)
        nn_angles = init_angles.repeat(self.K, 1).to(device).type(self.dtype)

        speed = init_velocities[0].repeat(self.K, 1).to(device)
        
        # Ensure poses tensor is on the correct device
        self.poses = self.poses.to(device)
        
        # Ensure noise tensor is on the correct device
        self.noise = self.noise.to(device)
        self.sigma = self.sigma.to(device)
        torch.normal(0, self.sigma, out=self.noise)
        
        # Ensure ctrl tensor is on the correct device
        self.ctrl = self.ctrl.to(device)
        
        # Loop the forward calculation till the horizon
        for t in range(self.T):
            ctrl_temp = (self.ctrl[t] + self.noise[t])  # Both tensors should now be on the same device
            ctrl_temp[:, 0].clamp_(self.min_speed, self.max_speed)
            ctrl_temp[:, 1].clamp_(self.min_steer_angle, self.max_steer_angle)
            
            # Calculate the change in pose
            delta_pose = MPPI6Planner.ackermann_bench(ctrl_temp[:, 0].unsqueeze(-1), ctrl_temp[:, 1].unsqueeze(-1), 
                                                    self.robot_length, dt)
            delta_pose = delta_pose.to(device)  # Ensure delta_pose is on the correct device
            
            cur_pose = MPPI6Planner.to_world_torch(init_pose_rep, delta_pose)
            self.poses[:, t, :] = cur_pose.clone()
            
            if torch.isnan(cur_pose).any():
                print("NaN is in poses")

            if self.times_mppi_called == 1 or (t == 0 and self.times_mppi_called % 50 == 0):
                self.images_for_poses = (self.terrain_manager.get_cropped_map_torch(self.poses[:, t, :], 8, self.K).unsqueeze(1) / self.terrain_manager.max_terrain_height).to(device)

            nn_input_norm = (nn_velocities - self.stats['velocity_mean']) / (self.stats['velocity_std'] + 0.000006)
            nn_angles_norm = (nn_angles - self.stats['roll_pitch_yaw_mean']) / (self.stats['roll_pitch_yaw_std'] + 0.000006)
            ctrl_temp_norm = (ctrl_temp - self.stats['cmd_vel_mean']) / (self.stats['cmd_vel_std'] + 0.000006) 
            
            # Model query for next pose calculation
            with torch.no_grad():
                model_otpt = self.kr_model(nn_input_norm.squeeze(1), nn_angles_norm, self.images_for_poses, ctrl_temp_norm)
            
            roll = (model_otpt[:, 0] * self.stats['res_roll_pitch_std'][0] + 0.000006) + self.stats['res_roll_pitch_mean'][0]
            pitch = (model_otpt[:, 1] * self.stats['res_roll_pitch_std'][1] + 0.000006) + self.stats['res_roll_pitch_mean'][1]

            # Ensure tensors are on the same device before assignment
            roll = roll.to(device)
            pitch = pitch.to(device)
            
            self.poses[:, t, [3]] = roll.unsqueeze(1)
            self.poses[:, t, [4]] = pitch.unsqueeze(1)
            change_in_robot = MPPI6Planner.to_robot_torch(init_pose_rep, self.poses[:, t, :])
            init_pose_rep = self.poses[:, t, :].clone()

            nn_velocities = change_in_robot / dt
            nn_angles[:, 2] = self.poses[:, t, 5]

            # Calculate the cost for each pose
            self.goal_tensor = self.goal_tensor.to(device)
            self.ctrl[t] = self.ctrl[t].to(device)
            self.noise[t] = self.noise[t].to(device)
            self.cost(self.poses[:, t, :], self.goal_tensor, self.ctrl[t], self.noise[t], roll, pitch, t, pure_pose)

        # MPPI weighing
        self.running_cost = self.running_cost.cpu()
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self._lambda
        torch.exp(self.running_cost, out=self.running_cost)
        weights = self.running_cost / torch.sum(self.running_cost)

        weights = weights.unsqueeze(1).expand(self.T, self.K, 2)
        weights_temp = weights.mul(self.noise.cpu())
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        
        # Move ctrl_change to the same device as ctrl before addition
        self.ctrl_change = self.ctrl_change.to(device)
        self.ctrl += self.ctrl_change

        self.ctrl[:,0].clamp_(self.min_speed, self.max_speed)
        self.ctrl[:,1].clamp_(self.min_steer_angle, self.max_steer_angle)
        
        return self.poses

    def odom_cb(self, vehicle_manager, m_system):
        """
        Callback to update the robot's pose using Chrono's simulation data.
        """
        timenow = m_system.GetChTime() 

        pos = vehicle_manager.get_position()
        euler_angles = vehicle_manager.get_rotation()
        roll = euler_angles.x
        pitch = euler_angles.y
        yaw = euler_angles.z

        # Update current pose first
        self.robot_pose = torch.Tensor([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw    
        ]).type(self.dtype)
        
        self.curr_pose = torch.Tensor([
                    pos.x, pos.y, pos.z,
                    roll, pitch, yaw
                ]).type(self.dtype)

        self.velocities = torch.Tensor([
                vehicle_manager.vehicle.GetVehicle().GetSpeed(), 0.0, 0.0,  # vel_x, vel_y, vel_z
                vehicle_manager.vehicle.GetVehicle().GetRollRate(),
                vehicle_manager.vehicle.GetVehicle().GetPitchRate(),
                vehicle_manager.vehicle.GetVehicle().GetYawRate()]).type(self.dtype)
        
        # Then handle initialization
        if self.last_pose is None:
            self.last_pose = torch.Tensor([
                pos.x, pos.y, pos.z,
                roll, pitch, yaw
            ]).type(self.dtype)
            self.lasttime = timenow
            return

        difference_from_goal = np.sqrt(((self.curr_pose.cpu())[0] - (self.goal_tensor.cpu())[0])**2 + ((self.curr_pose.cpu())[1] - (self.goal_tensor.cpu())[1])**2)
        # Adjust velocity limits based on distance to goal
        if difference_from_goal < 2:
            self.min_vel = -1  # Adjusted limits
            self.max_vel = 1
        else:
            self.min_vel = MIN_VEL
            self.max_vel = MAX_VEL

       # Update pose_dot if enough time has elapsed
        t_diff = timenow - self.last_t
        if t_diff >= 1:
            self.pose_dot = self.curr_pose - self.last_pose
            self.pose_dot[5] = self.util.clamp_angle(self.pose_dot[5])  # Clamp yaw difference
            self.last_pose = self.curr_pose
            self.last_t = timenow

    def mppi_cb(self):
        if self.curr_pose is None or self.goal_tensor is None or self.velocities is None:
            return
        
        roll, pitch, yaw = self.curr_pose[3], self.curr_pose[4], self.curr_pose[5]
        cs_angles = torch.Tensor([roll, pitch, yaw]).type(self.dtype)
        poses = self.mppi(self.curr_pose, self.velocities, cs_angles) 

        run_ctrl = None
        if not self.cont_ctrl:
            run_ctrl = self.get_control().cpu().numpy()
            self.recent_controls[self.control_i] = run_ctrl
            self.control_i = (self.control_i + 1) % self.recent_controls.shape[0]
            pub_control = self.recent_controls.mean(0)
            self.speed = pub_control[0]
            self.steering_angle = pub_control[1]

    def send_controls(self):
        """
        Sends control commands to the Chrono vehicle.
        :param vehicle: The Chrono HMMWV_Reduced vehicle object
        :param delta_steer: The steering angle from MPPI
        """
        if not self.at_goal:
            if self.cont_ctrl:  # Check if continuous control is enabled
                run_ctrl = self.get_control()
                if self.prev_ctrl is None:
                    self.prev_ctrl = run_ctrl

                # Update speed and steering
                speed = run_ctrl[0]  # Throttle value
                steer = -float(run_ctrl[1])  # Ensure steer is a float
                self.prev_ctrl = (speed, steer)
            else:
                speed = self.speed
                steer = -float(self.steering_angle)  # Ensure delta_steer is a float
        else:
            speed = 0.0
            steer = 0.0

        return speed, steer
    
    @staticmethod
    #function to find the change in robot frame given two poses
    def to_robot_torch(pose_batch1, pose_batch2):
        if pose_batch1.shape != pose_batch2.shape:
            raise ValueError("Input tensors must have same shape")

        if pose_batch1.shape[-1] != 6:
            raise ValueError(f"Input tensors must have last dim equal to 6 for SE3, got {pose_batch1.shape[-1]}")

        batch_size = pose_batch1.shape[0]
        ones = torch.ones_like(pose_batch2[:, 0])
        transform = torch.zeros_like(pose_batch1)
        T1 = torch.zeros((batch_size, 4, 4), device=pose_batch1.device, dtype=pose_batch1.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=pose_batch2.device, dtype=pose_batch2.dtype)

        T1[:, :3, :3] = MPPI6Planner.euler_to_rotation_matrix(pose_batch1[:, 3:])
        T2[:, :3, :3] = MPPI6Planner.euler_to_rotation_matrix(pose_batch2[:, 3:])
        T1[:, :3,  3] = pose_batch1[:, :3]
        T2[:, :3,  3] = pose_batch2[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1

        T1_inv = torch.inverse(T1)
        tf3_mat = torch.matmul(T2, T1_inv)

        transform[:, :3] = torch.matmul(T1_inv, torch.cat((pose_batch2[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
        transform[:, 3:] = MPPI6Planner.extract_euler_angles_from_se3_batch(tf3_mat)

        return transform
         
    @staticmethod 
    def to_world_torch(Robot_frame, P_relative):
        SE3 = True

        if not isinstance(Robot_frame, torch.Tensor):
            Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
        if not isinstance(P_relative, torch.Tensor):
            P_relative = torch.tensor(P_relative, dtype=torch.float32)

        if len(Robot_frame.shape) == 1:
            Robot_frame = Robot_frame.unsqueeze(0)

        if len(P_relative.shape) == 1:
            P_relative = P_relative.unsqueeze(0)
    
        if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
            raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")

        # pdb.set_trace()
        if Robot_frame.shape != P_relative.shape:
            raise ValueError("Input tensors must have same shape")
        
        if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
            raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
        
        if Robot_frame.shape[-1] == 3:
            SE3 = False
            Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
            Robot_frame_[:, [0,1,5]] = Robot_frame
            Robot_frame = Robot_frame_
            P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
            P_relative_[:, [0,1,5]] = P_relative
            P_relative = P_relative_
            
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

        R1 = MPPI6Planner.euler_to_rotation_matrix(Robot_frame[:, 3:])
        R2 = MPPI6Planner.euler_to_rotation_matrix(P_relative[:, 3:])
        
        T1[:, :3, :3] = R1
        T2[:, :3, :3] = R2
        T1[:, :3,  3] = Robot_frame[:, :3]
        T2[:, :3,  3] = P_relative[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 

        T_tf = torch.matmul(T2, T1)
        transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
        transform[:, 3:] = MPPI6Planner.extract_euler_angles_from_se3_batch(T_tf)

        if not SE3:
            transform = transform[:, [0,1,5]]

        return transform
           
    @staticmethod 
    def euler_to_rotation_matrix(euler_angles):
        """ Convert Euler angles to a rotation matrix """
        # Compute sin and cos for Euler angles
        cos = torch.cos(euler_angles)
        sin = torch.sin(euler_angles)
        zero = torch.zeros_like(euler_angles[:, 0])
        one = torch.ones_like(euler_angles[:, 0])
        # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
        R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
        R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
        R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

        return torch.matmul(torch.matmul(R_z, R_y), R_x)

    @staticmethod 
    def extract_euler_angles_from_se3_batch(tf3_matx):
        # Validate input shape
        if tf3_matx.shape[1:] != (4, 4):
            raise ValueError("Input tensor must have shape (batch, 4, 4)")

        # Extract rotation matrices
        rotation_matrices = tf3_matx[:, :3, :3]

        # Initialize tensor to hold Euler angles
        batch_size = tf3_matx.shape[0]
        euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

        # Compute Euler angles
        euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
        euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
        euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

        return euler_angles

    @staticmethod
    def ackermann_bench(throttle, steering, wheel_base=2.85, dt=0.1):
        if not isinstance(throttle, torch.Tensor):
            throttle = torch.tensor(throttle, dtype=torch.float32)
        if not isinstance(steering, torch.Tensor):
            steering = torch.tensor(steering, dtype=torch.float32)
        if not isinstance(wheel_base, torch.Tensor):
            wheel_base = torch.tensor(wheel_base, dtype=torch.float32)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, dtype=torch.float32)
        if throttle.shape != steering.shape:
            raise ValueError("throttle and steering must have the same shape")
        if len(throttle.shape) == 0:
            throttle = throttle.unsqueeze(0)
        
        deltaPose = torch.zeros(throttle.shape[0], 6, dtype=torch.float32)

        dtheta = (throttle / wheel_base) * torch.tan(steering) * dt
        dx = throttle * torch.cos(dtheta) * dt
        dy = throttle * torch.sin(dtheta) * dt
        deltaPose[:, 0], deltaPose[:, 1], deltaPose[:, 5] = dx.squeeze(), dy.squeeze(), dtheta.squeeze()

        return deltaPose.squeeze()
    
    def bernstein_poly(self, i, n, t):
        """Calculate Bernstein polynomial basis"""
        return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

    def bezier_curve(self, points, num_samples=100):
        """Create a Bezier curve from control points"""
        n = len(points) - 1
        t = np.linspace(0, 1, num_samples)
        curve = np.zeros((num_samples, 2))
        for i in range(n + 1):
            curve += np.outer(self.bernstein_poly(i, n, t), points[i])
        return curve

    def smooth_path_bezier(self, path, num_samples=100):
        """Smooth path using Bezier curves"""
        if path is None or len(path) < 3:
            return path
        
        points = np.array(path)
        curve = self.bezier_curve(points, num_samples)
        return curve
        
    def chrono_to_grid(self, pos, inflated_map, m_terrain_length, m_terrain_width):
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
        
    def astar_path(self, obs_path, start_pos, goal_pos):
        """Generate initial A* path"""
        m_terrain_length = self.terrain_manager.terrain_length
        m_terrain_width = self.terrain_manager.terrain_width
        
        # Load obstacle map
        obs_image = Image.open(obs_path)
        obs_map = np.array(obs_image.convert('L'))
        grid_map = np.where(obs_map == 255, 1, 0)
        structure = np.ones((2 * self.inflation_radius + 1, 2 * self.inflation_radius + 1))
        inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
        
        # Convert start and goal to grid coordinates
        start_grid = self.chrono_to_grid(start_pos, inflated_map, m_terrain_length, m_terrain_width)
        goal_grid = self.chrono_to_grid(goal_pos, inflated_map, m_terrain_length, m_terrain_width)
        
        # Plan path
        planner = AStarPlanner(inflated_map, start_grid, goal_grid)
        path = planner.plan()
        path = self.smooth_path_bezier(path)
        
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
        #            extent=[0, 129 * self.terrain_manager.scale_factor, 129 * self.terrain_manager.scale_factor, 0])
        
        # # Plot the smooth path with professional styling
        # path_y = [p[0] * self.terrain_manager.scale_factor for p in path]
        # path_x = [p[1] * self.terrain_manager.scale_factor for p in path]
        # plt.plot(path_x, path_y, color='#D62728', linewidth=2, zorder=3)

        # # Plot start and goal with consistent styling
        # plt.scatter(start_grid[1] * self.terrain_manager.scale_factor, start_grid[0] * self.terrain_manager.scale_factor, 
        #             color='green', s=150, zorder=4, edgecolor='green', linewidth=1.5)
        # plt.scatter(goal_grid[1] * self.terrain_manager.scale_factor, goal_grid[0] * self.terrain_manager.scale_factor, 
        #             color='red', s=200, marker='*', zorder=4, edgecolor='darkred', linewidth=1.5)

        # # Set axis limits
        # plt.xlim(0, 129 * self.terrain_manager.scale_factor)
        # plt.ylim(129 * self.terrain_manager.scale_factor, 0)
        
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
        chrono_path = self.terrain_manager.transform_to_chrono(bitmap_points)
        
        return chrono_path
        
    def astar_replan(self, obs_path, current_pos, goal_pos):
        """Replan path from current position to goal"""
        m_terrain_length = self.terrain_manager.terrain_length
        m_terrain_width = self.terrain_manager.terrain_width
        
        # Load obstacle map
        obs_image = Image.open(obs_path)
        obs_map = np.array(obs_image.convert('L'))
        grid_map = np.where(obs_map == 255, 1, 0)
        structure = np.ones((2 * self.inflation_radius + 1, 2 * self.inflation_radius + 1))
        inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
        
        # Convert current pos and goal to grid coordinates
        current_grid = self.chrono_to_grid(current_pos, inflated_map, m_terrain_length, m_terrain_width)
        goal_grid = self.chrono_to_grid(goal_pos, inflated_map, m_terrain_length, m_terrain_width)
        
        # Replan path
        planner = AStarPlanner(inflated_map, current_grid, goal_grid)
        path = planner.plan()
        
        if path is None:
            print("No valid path found in replanning!")
            return None
        
        if len(path) >= 3:
            path = self.smooth_path_bezier(path)
            
        # Convert to Chrono coordinates
        bitmap_points = [(point[1], point[0]) for point in path]
        chrono_path = self.terrain_manager.transform_to_chrono(bitmap_points)
        
        return chrono_path
    
    def find_local_goal(self, vehicle_pos, vehicle_heading, chrono_path, local_goal_idx):
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
            
            if distance >= self.look_ahead_distance:
                dx = path_point[0] - vehicle_pos[0]
                dy = path_point[1] - vehicle_pos[1]
                angle_to_point = np.arctan2(dy, dx)
                angle_diff = (angle_to_point - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) <= np.pi / 2:
                    return idx, path_point
                
        final_idx = len(chrono_path) - 1
        return final_idx, chrono_path[final_idx]

    def compute_throttle(self, desired_speed, time, step_size, vehicle_ref_frame):
        """Compute throttle and braking inputs"""
        if self.speed_controller is None:
            # Initialize speed controller
            self.speed_controller = veh.ChSpeedController()
            self.speed_controller.Reset(vehicle_ref_frame)
            self.speed_controller.SetGains(1.0, 0.0, 0.0)
            
        # Compute throttle/braking
        out_throttle = self.speed_controller.Advance(vehicle_ref_frame, desired_speed, time, step_size)
        out_throttle = np.clip(out_throttle, -1, 1)
        
        # Split into throttle and braking
        if out_throttle > 0:
            return out_throttle, 0.0
        else:
            return 0.0, -out_throttle
