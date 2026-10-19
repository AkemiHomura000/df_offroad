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

class TNTPlanner:
    def __init__(self, terrain_manager):
        #Robot Limits
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.max_del = 1
        self.min_del = -1
        self.robot_length = 0.54
        self.sampled_trajectories = []
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
        self.K = 800                     # Number of sample rollouts
        self.dt = 1
        self._lambda = 0.1               # Temperature
        self.sigma = torch.Tensor([0.2, 0.5]).type(torch.float32).expand(self.T, self.K, 2)  # (T, K, 2)
        self.inv_sigma = 1 / self.sigma[0, 0, :]  # (2, )
        
        stats_pickle_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'stats_rock.pickle')
        with open(stats_pickle_path, 'rb') as f:
            scale = pickle.load(f)
        
        self.scale_state = scale['pose_dot']
        self.offset_scale = scale['map_offset']

        self.robot_pose = None                                                                    # Robot pose
        self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype)                               # (T,K,2)
        self.poses = torch.Tensor(self.K, self.T, 6).type(self.dtype)                               # (K,T,6)
        self.fpose = torch.Tensor(self.K, 6).type(self.dtype)                                    # (K,6)
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
        self.running_cost = torch.zeros(self.K).type(self.dtype)                                    # (K, )
        self.pose_cost = torch.Tensor(self.K).type(self.dtype)                                      # (K, )
        self.bounds_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )
        self.height_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )#ony for crawler
        self.ctrl_cost = torch.Tensor(self.K, 2).type(self.dtype)                                   # (K,2)
        self.ctrl_change = torch.Tensor(self.T,2).type(self.dtype)                                  # (T,2)
        self.euclidian_distance = torch.Tensor(self.K).type(self.dtype)                             # (K, )
        self.dist_to_goal = torch.Tensor(self.K, 6).type(self.dtype)                                # (K, )
        
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
        self.chrono_path = None

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
        self.traversability_model(torch.zeros(1, 1, 160, 130).to(self.device))

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
                map_t = F.center_crop(map_d, (320, 260))
                map_t = F.gaussian_blur(map_t, kernel_size=3, sigma=0.2)
                map_t = F.resize(map_t, (160, 130))
                traversability_maps = self.traversability_model(map_t).squeeze()
                self.map_embedding = self.model.process_map(map_d).repeat(self.K, 1, 1, 1).cuda()
                
                # Combine traversability maps
                traversability_maps = traversability_maps[self.m_idx]
                traversability_maps = traversability_maps * self.weights
                combined_traversability_map = torch.sum(traversability_maps, dim=0).squeeze()

                elevmap_n = F.center_crop(map_d, (320, 260)).squeeze()
                combined_traversability_map[abs(elevmap_n - 0.8) > 0.425] = combined_traversability_map.max()*10
                map_image = cv2.rotate(combined_traversability_map.cpu().numpy(), cv2.ROTATE_90_COUNTERCLOCKWISE)
                # cv2.imshow("Traversability Map", map_image)
                
                # cv2.waitKey(1)

                self.trav_map_combined = combined_traversability_map

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
    
    def cost(self, pose, goal, ctrl, noise, t):
        self.obstacle_map = self.obstacle_map.to(self.device)
        self.fpose.copy_(pose)
        self.dist_to_goal.copy_(self.fpose).sub_(goal)
        
        self.dist_to_goal[:,3] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,3])
        self.dist_to_goal[:,4] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,4])
        self.dist_to_goal[:,5] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,5])

        xy_to_goal = self.dist_to_goal[:,:2]
        self.euclidian_distance = torch.norm(xy_to_goal, p=2, dim=1)
        euclidian_distance_squared = self.euclidian_distance.pow(2)

        self.ctrl_cost.copy_(ctrl).mul_(self._lambda).mul_(self.inv_sigma).mul_(noise).mul_(0.5)
        running_cost_temp = self.ctrl_cost.abs_().sum(dim=1)
        self.running_cost.copy_(running_cost_temp)
        
        # Get map dimensions
        map_height, map_width = self.trav_map_combined.size()

        # Use vehicle's actual position for traversability assessment
        px = (map_height // 2 - pose[:,1]).long().clone().detach().to(self.device)
        py = (map_width // 2 + pose[:,0]).long().clone().detach().to(self.device)
        
        px[px<=0] = 0
        px[px>=map_width] = map_width - 1
        py[py<=0] = 0
        py[py>=map_height] = map_height -1

        # Calculate traversability cost
        low_traversability_penalty = self.trav_map_combined[py, px].to(self.running_cost.device)

        map_o_height, map_o_width = self.obstacle_map.size()

        # Use vehicle's actual position for traversability assessment
        p_x = (map_o_height // 2 - pose[:,1]).long().clone().detach().to(self.device)
        p_y = (map_o_width // 2 + pose[:,0]).long().clone().detach().to(self.device)
        
        p_x[p_x<=0] = 0
        p_x[p_x>=map_o_width] = map_o_width - 1
        p_y[p_y<=0] = 0
        p_y[p_y>=map_o_height] = map_o_height -1

        obstacle_penalty = (self.obstacle_map[p_y, p_x]==255).float().to(self.running_cost.device)
        
        # Normalize costs
        eu_min = euclidian_distance_squared.min()
        eu_max = euclidian_distance_squared.max()
        euclidian_distance_squared = (euclidian_distance_squared - eu_min) / (eu_max + 1e-6 - eu_min)

        lt_min = low_traversability_penalty.min()
        lt_max = low_traversability_penalty.max()
        low_traversability_penalty = (low_traversability_penalty - lt_min) / (lt_max + 1e-6 - lt_min)

        # Add all costs
        self.running_cost.add_(euclidian_distance_squared*5).add_(low_traversability_penalty*3).add_(obstacle_penalty*4)

    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone()

        # shift all controls forward by 1, with last control replicated
        self.ctrl = torch.roll(self.ctrl, shifts=-1, dims=0)
        return run_ctrl

    def mppi(self, init_pose, init_inp):
        # init_pose (6, ) [x, y, z, r, p, y]
        
        # init_input (17,):
        #   0    1      2     3     4     5       6          7          8           9         10        11       12    13    14   15     16     
        # xdot, ydot, zdot, rdot, pdot, ywdot, sin(roll), cos(roll), sin(pitch), cos(pitch), sin(yaw), cos(yaw), vel, delta, dt, map_1, map_2
        
        t0 = time.time()
        dt = self.dt

        self.running_cost.zero_()                                                  # Zero running cost
        pose = init_pose.repeat(self.K, 1).cuda()                                  # Repeat the init pose to sample size 
        nn_input = init_inp.repeat(self.K, 1).cuda()                               # Repeat the init input to sample size
        
        # Initialize storage for this rollout's trajectories
        current_rollout = torch.zeros(self.K, self.T, 6).cuda()

        state = nn_input[:, :6]                                                    # Get the state from the input
        cmd_vel = nn_input[:, 6:8]                                                 # Get the control from the input
        map_offset = torch.zeros(self.K, 4).cuda()                                 # Get the map offset from the input
        map_offset[:, :2] = nn_input[:, 8:10]                                      # Get the map offset from the input
        map_offset[:, 2] = torch.sin(pose[:, 5])
        map_offset[:, 3] = torch.cos(pose[:, 5])

        elev_map = self.map_embedding                                              # Repeat the map embedding to sample size
        torch.normal(0, self.sigma, out=self.noise)                                # Generate noise based on the sigma
        if not wm_vct:
            state[:,[0,1,2]] = state[:,[0,1,2]] * 0.1
            state = self.util.scale_in(state, self.scale_state, 0)
            map_offset = self.util.scale_in(map_offset, self.offset_scale, 1)

        # Loop the forward calculation till the horizon
        for t in range(self.T):
            cmd_vel = (self.ctrl[t] + self.noise[t]).cuda() 
            # noise_scale = max(0.3, 1.0 - t/self.T)                                 # Reduce noise over horizon
            # self.noise[t] *= noise_scale                                           # Add noise to previous control input
            cmd_vel[:, 0].clamp_(self.min_vel, self.max_vel)                       # Clamp control velocity
            cmd_vel[:, 1].clamp_(self.min_del, self.max_del)                       # Clamp control steering

            # Model query for next pose caalculation
            with torch.no_grad():
                if not wm_vct:
                    out = self.model.predict(state, cmd_vel, map_offset, elev_map)
                else:
                    out = self.util.ackermann_model(cmd_vel)
            state = out
            model_output = out.detach().clone()
            
            # Scale the output to add it in pose
            if not wm_vct:
                se2_pose = pose[:, [0,1,5]].clone()
                model_out_scaled = self.util.scale_out(model_output.clone(), self.scale_state, 0)
                pose_temp, state = self.util.get_next_batch_se2(model_output, se2_pose, self.scale_state)
                
                pose[:, [0,1]] = pose_temp[:,[0,1]] 
                pose[:, 2] += model_out_scaled[:, 2]
                pose[:, 3:5] = model_out_scaled[:, 3:5] 
                map_offset = self.util.get_next_offsets_se2(map_offset, se2_pose, pose_temp, self.offset_scale)

            else:
                se2_pose = pose[:, [0,1,5]].clone()
                pose_temp, state = self.util.get_next_batch_se2(model_output, se2_pose, self.scale_state)
                pose[:, [0,1,5]] = pose_temp

            # Add to self poses
            self.poses[:,t,:] = pose.clone()
            current_rollout[:, t, :] = pose.clone()

            self.sampled_trajectories = current_rollout.cpu().numpy()

            # Calculate the cost for each pose
            self.cost(pose, self.goal_tensor, self.ctrl[t], self.noise[t], t)

        # MPPI weighing
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self._lambda
        torch.exp(self.running_cost, out=self.running_cost)
        weights = self.running_cost / torch.sum(self.running_cost)+1e-6

        weights = weights.unsqueeze(1).expand(self.T, self.K, 2)
        weights_temp = weights.mul(self.noise)
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        self.ctrl += self.ctrl_change
        self.ctrl[:,0].clamp_(self.min_vel, self.max_vel)
        self.ctrl[:,1].clamp_(self.min_del, self.max_del)

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
        if t_diff >= 0.1:
            self.pose_dot = (self.curr_pose - self.last_pose)
            self.pose_dot[5] = self.util.clamp_angle(self.pose_dot[5])  # Clamp yaw difference
            self.last_pose = self.curr_pose
            self.last_t = timenow

    def mppi_cb(self, curr_pose, pose_dot):
        if curr_pose is None or self.goal_tensor is None:
            return
        
        roll, pitch, yaw = (self.curr_pose.cpu())[3], (self.curr_pose.cpu())[4], (self.curr_pose.cpu())[5]
        pose_dot[:3] = torch.clamp(pose_dot[:3], -self.scale_state[0], self.scale_state[0])
        pose_dot[5] = torch.clamp(pose_dot[5], -self.scale_state[2], self.scale_state[2])

        nn_input = torch.Tensor([pose_dot[0], pose_dot[1], pose_dot[2], roll, pitch, pose_dot[5],
                                0.0, 0.0, 0.1, 0.0, 0.0]).type(self.dtype)

        poses = self.mppi(curr_pose, nn_input)

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
                speed = (run_ctrl[0])  # Throttle value
                steer = -float(run_ctrl[1])  # Ensure steer is a float
                self.prev_ctrl = (speed, steer)
            else:
                speed = (self.speed)
                steer = -float(self.steering_angle)  # Ensure delta_steer is a float
        else:
            speed = 0.0
            steer = 0.0

        return speed, steer
    
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
