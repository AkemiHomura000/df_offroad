import numpy as np
import heapq
from PIL import Image
from scipy.ndimage import binary_dilation
from scipy.special import comb
import matplotlib.pyplot as plt

import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr

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

class EHPlanner:
    def __init__(self, terrain_manager):
        self.terrain_manager = terrain_manager
        self.look_ahead_distance = 10.0 * self.terrain_manager.scale_factor
        self.inflation_radius = int(4.0 * self.terrain_manager.scale_factor)
        
        # Controllers
        self.steering_controller = PIDController(kp=1.0, ki=0.0, kd=0.0)
        self.speed_controller = None 
        
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
    
    def compute_steering(self, heading_error, step_size):
        """Compute steering input using PID controller"""
        # PID controller to compute steering
        steering = -self.steering_controller.compute(heading_error, step_size)
        return steering
        
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
