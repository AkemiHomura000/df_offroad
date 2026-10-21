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
from verti_bench.systems.DDV2.DDV2 import DDV2Planner
from verti_bench.vehicles.HMMWV import HMMWVManager
from verti_bench.vehicles.FEDA import FEDAManager
from verti_bench.vehicles.Gator import GatorManager
from verti_bench.vehicles.MAN5t import MAN5tManager
from verti_bench.vehicles.MAN7t import MAN7tManager
from verti_bench.vehicles.MAN10t import MAN10tManager
from verti_bench.vehicles.M113 import M113Manager
from verti_bench.vehicles.ART import ARTManager
from verti_bench.vehicles.VW import VWManager

class DDV2Sim:
    """Verti-Bench sim with DiffusionDrive local planning (+ PID track)."""

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
        # DDV2 always needs sensors
        self.ddv2_sensors = True
        self.ddv2_dump_dir = config.get('ddv2_dump_dir', None)
        self.ddv2_host = config.get('ddv2_host', '127.0.0.1')
        self.ddv2_port = int(config.get('ddv2_port', 8765))
        self.ddv2_suite = None
        self._ddv2_last_sample_t = -1.0
        self._ddv2_dump_count = 0
        self.plan_client = None
        
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
        self.last_replan_time = -1e9
        self.replan_interval = float(config.get('ddv2_replan_interval', 0.5))
        self.vis = None
        self.driver = None
        self.chrono_path = None
        self.local_goal_idx = 0
        self.track_speed = self.speed
        self.ddv2_live_viz = bool(config.get("ddv2_live_viz", True))
        self.ddv2_draw_world = bool(config.get("ddv2_draw_world", True))
        self.ddv2_verbose = bool(config.get("ddv2_verbose", True))
        self._live_viz = None
        self._traj_markers = []
        self._infer_ms_hist = []
        self.termination_reason = None
        self.start_pos = None
        self.goal_pos = None
        
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
        """Connect to DDV2 plan server and build local tracker."""
        if self.system_type.lower() != 'ddv2':
            raise ValueError(f"Unsupported system type: {self.system_type}.")
        sys.path.insert(0, "/home/rm/Desktop/df_offroad/scripts/bridge")
        from ddv2_plan_client import DDV2PlanClient

        self.plan_client = DDV2PlanClient(host=self.ddv2_host, port=self.ddv2_port)
        self.plan_client.connect()
        self.planner = DDV2Planner(self.terrain_manager, self.plan_client)
        print(
            f"[ddv2] planner ready → {self.ddv2_host}:{self.ddv2_port} "
            f"replan every {self.replan_interval:.2f}s"
        )
    
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
        self.start_pos = list(start_pos)
        self.goal_pos = list(goal_pos)
        self.termination_reason = None
        
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
        
        # Local path filled on first successful DDV2 inference
        self.chrono_path = None
        self.local_goal_idx = 0
        
        # Set up visualization
        if self.render:
            self._setup_visualization()
            
        # Set up driver
        self._setup_driver()

        from verti_bench.envs.utils.ddv2_sensors import DDV2SensorSuite
        self.ddv2_suite = DDV2SensorSuite(
            self.system, self.vehicle_manager.chassis_body
        )
        if self.ddv2_verbose:
            print("DDV2 sensors enabled: front cam + lidar BEV → plan server")
        if self.ddv2_dump_dir:
            os.makedirs(self.ddv2_dump_dir, exist_ok=True)
        if self.ddv2_live_viz:
            try:
                from verti_bench.systems.DDV2.ddv2_live_viz import DDV2LiveViz

                save = None
                if self.ddv2_dump_dir:
                    save = os.path.join(self.ddv2_dump_dir, "live_panel.jpg")
                self._live_viz = DDV2LiveViz(save_path=save)
                if self.ddv2_verbose:
                    print("[ddv2] live viz: separate OpenCV window + live_panel.jpg")
            except Exception as e:
                print(f"[ddv2] live viz disabled: {e}")
                self._live_viz = None

    def _navsim_heading_rad(self) -> float:
        """Yaw of chassis +X in world XY — same frame as ddv2_sensors cmd/vel."""
        import math

        x_axis = self.vehicle_manager.chassis_body.GetRot().GetAxisX()
        return math.atan2(x_axis.y, x_axis.x)

    def _clear_traj_markers(self):
        for b in self._traj_markers:
            try:
                self.system.Remove(b)
            except Exception:
                pass
        self._traj_markers = []

    def _draw_traj_world(self, path, vehicle_pos):
        """Draw predicted waypoints as small spheres in Chrono (Irrlicht)."""
        if not self.ddv2_draw_world or not path:
            return
        self._clear_traj_markers()
        z0 = float(vehicle_pos.z) + 0.4
        for i, (x, y) in enumerate(path):
            r = 0.25 if i < len(path) - 1 else 0.4
            body = chrono.ChBodyEasySphere(r, 1000, True, True)
            body.SetPos(chrono.ChVector3d(float(x), float(y), z0))
            body.SetFixed(True)
            body.EnableCollision(False)
            # cyan-ish for mid, yellow for end
            if hasattr(body, "GetVisualModel") and body.GetVisualModel() is not None:
                pass
            try:
                mat = chrono.ChVisualMaterial()
                if i == len(path) - 1:
                    mat.SetDiffuseColor(chrono.ChColor(1.0, 0.85, 0.1))
                else:
                    mat.SetDiffuseColor(chrono.ChColor(0.1, 0.85, 0.95))
                for sh in body.GetVisualModel().GetShapes():
                    sh.SetMaterial(0, mat)
            except Exception:
                pass
            self.system.Add(body)
            self._traj_markers.append(body)

    def run(self):
        """Run the simulation"""
        # Check initialization
        if not hasattr(self, 'terrains') or not self.terrains:
            raise ValueError("Simulation not initialized. Call initialize() first.")

        try:
            return self._run_loop()
        finally:
            if self.plan_client is not None:
                self.plan_client.close()
            self._clear_traj_markers()
            if self._live_viz is not None:
                self._live_viz.close()

    def _run_loop(self):
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
                    # Pump OpenCV panel every Irrlicht frame (otherwise window never maps)
                    if self._live_viz is not None:
                        self._live_viz.pump()
            
            # Get vehicle position and orientation
            vehicle_pos = self.vehicle_manager.get_position()
            vector_to_goal = self.vehicle_manager.goal - vehicle_pos
            
            # Sensors before planning
            if self.ddv2_suite is not None:
                self.ddv2_suite.update()

            # Update controls
            if self.use_gui:
                self.driver_inputs = self.driver.GetInputs()
            else:
                euler_angles = self.vehicle_manager.get_rotation()
                roll = euler_angles.x
                pitch = euler_angles.y
                # NAVSIM/DDV2 heading: XY projection of chassis +X (fwd).
                # Do NOT use Cardan Z here — with roll/pitch it disagrees with
                # ddv2_sensors._driving_command / vel_xy and can mirror L/R in world.
                vehicle_heading = self._navsim_heading_rad()
                roll_angles.append(np.degrees(abs(roll)))
                pitch_angles.append(np.degrees(abs(pitch)))

                need_replan = (
                    self.chrono_path is None
                    or (time - self.last_replan_time) >= self.replan_interval
                )
                if need_replan and self.ddv2_suite is not None:
                    obs = self.ddv2_suite.get_observation(goal=self.vehicle_manager.goal)
                    if obs is not None:
                        try:
                            self.chrono_path = self.planner.plan_local(
                                obs,
                                (vehicle_pos.x, vehicle_pos.y),
                                vehicle_heading,
                            )
                            self.local_goal_idx = 0
                            self.track_speed = self.planner.suggested_speed(self.speed)
                            self.last_replan_time = time
                            infer_ms = self.planner.last_infer_ms
                            rtt_ms = self.planner.last_rtt_ms
                            if np.isfinite(infer_ms):
                                self._infer_ms_hist.append(infer_ms)
                            self._draw_traj_world(self.chrono_path, vehicle_pos)
                            if self._live_viz is not None:
                                self._live_viz.update(
                                    obs=obs,
                                    traj_ego=self.planner.last_traj,
                                    infer_ms=infer_ms,
                                    rtt_ms=rtt_ms,
                                    sim_t=time,
                                    track_speed=self.track_speed,
                                    vehicle_xy=(vehicle_pos.x, vehicle_pos.y),
                                )
                            if self.ddv2_dump_dir and self._ddv2_dump_count < 5:
                                path = os.path.join(
                                    self.ddv2_dump_dir, f"obs_{self._ddv2_dump_count:03d}.npz"
                                )
                                np.savez_compressed(
                                    path,
                                    camera_feature=obs["camera_feature"],
                                    lidar_feature=obs["lidar_feature"],
                                    status_feature=obs["status_feature"],
                                    trajectory=self.planner.last_traj,
                                    infer_ms=np.array([infer_ms], dtype=np.float32),
                                    rtt_ms=np.array([rtt_ms], dtype=np.float32),
                                )
                                self._ddv2_dump_count += 1
                            mean_ms = (
                                float(np.mean(self._infer_ms_hist))
                                if self._infer_ms_hist
                                else float("nan")
                            )
                            from verti_bench.systems.DDV2.ddv2_live_viz import parse_nav_status

                            _mode, nav_label, _nav = parse_nav_status(obs["status_feature"])
                            te = self.planner.last_traj[-1] if self.planner.last_traj is not None else None
                            # NAVSIM: y>0 left, y<0 right — flag cmd/traj disagreement
                            side = "?"
                            if te is not None:
                                if te[1] > 0.3:
                                    side = "trajLEFT(+y)"
                                elif te[1] < -0.3:
                                    side = "trajRIGHT(-y)"
                                else:
                                    side = "trajSTRAIGHT"
                            if self.ddv2_verbose:
                                print(
                                    f"[ddv2] t={time:.2f}s  {nav_label}  {side}  "
                                    f"GPU={infer_ms:.1f}ms RTT={rtt_ms:.1f}ms "
                                    f"(μ_GPU={mean_ms:.1f} n={len(self._infer_ms_hist)})  "
                                    f"ego_end=({te[0]:.1f},{te[1]:.1f})  "
                                    f"world_end=({self.chrono_path[-1][0]:.1f},{self.chrono_path[-1][1]:.1f})  "
                                    f"v≈{self.track_speed:.2f}"
                                )
                        except Exception as e:
                            print(f"[ddv2] plan failed: {e}")

                if self.chrono_path is None:
                    # Warm up sensors / wait for first plan
                    self.driver_inputs.m_steering = 0.0
                    self.driver_inputs.m_throttle = 0.0
                    self.driver_inputs.m_braking = 0.2
                else:
                    self.local_goal_idx, local_goal = self.planner.find_local_goal(
                        (vehicle_pos.x, vehicle_pos.y),
                        vehicle_heading,
                        self.chrono_path,
                        self.local_goal_idx,
                    )
                    steering = self.planner.compute_steering(
                        vehicle_heading,
                        local_goal,
                        (vehicle_pos.x, vehicle_pos.y),
                        self.step_size,
                    )
                    self.driver_inputs.m_steering = np.clip(
                        steering,
                        self.driver_inputs.m_steering - 0.05,
                        self.driver_inputs.m_steering + 0.05,
                    )
                    throttle, braking = self.planner.compute_throttle(
                        self.track_speed,
                        time,
                        self.step_size,
                        self.vehicle_manager.vehicle.GetVehicle().GetRefFrame(),
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
                    if self.ddv2_verbose:
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
                    self.termination_reason = 'stuck'
                    return time - start_time, False, avg_roll, avg_pitch
            
            self.last_position = current_position
            
            # Check if goal reached
            if vector_to_goal.Length() < 8 * self.scale_factor:
                if self.ddv2_verbose:
                    print('--------------------------------------------------------------')
                    print('Goal Reached')
                    print(f'Initial position: {self.vehicle_manager.init_loc}')
                    print(f'Goal position: {self.vehicle_manager.goal}')
                    print('--------------------------------------------------------------')
                
                if self.render:
                    self.vis.Quit()
                    
                avg_roll = np.mean(roll_angles) if roll_angles else 0 
                avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
                self.termination_reason = 'success'
                return time - start_time, True, avg_roll, avg_pitch
            
            # Check if time limit exceeded
            if time > self.max_time:
                if self.ddv2_verbose:
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
                self.termination_reason = 'timeout'
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

        return None, False, 0, 0
