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

from verti_bench.envs.utils.utils import SetChronoDataDirectories
from verti_bench.envs.utils.ros_msg_stubs import ensure_ros_msg_stubs

# Lazy imports: MPPI/TAL/TNT/etc. need ROS grid_map_msgs; DDV2/PID do not.
# Offline stubs cover missing ROS on headless collect boxes.
_SYSTEM_SIM = {
    'pid': ('verti_bench.systems.PID.PID_sim', 'PIDSim'),
    'eh': ('verti_bench.systems.EH.EH_sim', 'EHSim'),
    'mppi': ('verti_bench.systems.MPPI.MPPI_sim', 'MPPISim'),
    'rl': ('verti_bench.systems.RL.RL_sim', 'RLSim'),
    'mcl': ('verti_bench.systems.MCL.MCL_sim', 'MCLSim'),
    'acl': ('verti_bench.systems.ACL.ACL_sim', 'ACLSim'),
    'wmvct': ('verti_bench.systems.WMVCT.WMVCT_sim', 'WMVCTSim'),
    'mppi6': ('verti_bench.systems.MPPI6.MPPI6_sim', 'MPPI6Sim'),
    'tal': ('verti_bench.systems.TAL.TAL_sim', 'TALSim'),
    'tnt': ('verti_bench.systems.TNT.TNT_sim', 'TNTSim'),
    'manual': ('verti_bench.systems.Manual.Manual_sim', 'ManualSim'),
    'ddv2': ('verti_bench.systems.DDV2.DDV2_sim', 'DDV2Sim'),
}


def _load_sim_class(system):
    if system not in _SYSTEM_SIM:
        raise ValueError(f"Unknown system: {system}")
    if system in ('mppi', 'mppi6', 'tal', 'tnt', 'wmvct'):
        ensure_ros_msg_stubs()
    module_name, class_name = _SYSTEM_SIM[system]
    import importlib
    return getattr(importlib.import_module(module_name), class_name)


def single_experiment(config):
    """Run a single simulation experiment"""
    SimCls = _load_sim_class(config['system'])
    sim = SimCls(config)
    sim.initialize()
    
    # Run simulation
    time_to_goal, success, avg_roll, avg_pitch = sim.run()
    
    # Return results
    return {
        'time_to_goal': time_to_goal if success else None,
        'success': success,
        'avg_roll': avg_roll,
        'avg_pitch': avg_pitch
    }

def multiple_experiments(config, num_experiments=5):
    """Run multiple simulation experiments and aggregate results"""
    results = []
    
    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        result = single_experiment(config)
        results.append(result)
        
    # Process results 
    success_count = sum(1 for r in results if r['success'])
    successful_times = [r['time_to_goal'] for r in results if r['time_to_goal'] is not None]
    avg_rolls = [r['avg_roll'] for r in results if r['success']]
    avg_pitches = [r['avg_pitch'] for r in results if r['success']]

    mean_traversal_time = np.mean(successful_times) if successful_times else None
    roll_mean = np.mean(avg_rolls) if avg_rolls else None
    roll_variance = np.var(avg_rolls) if avg_rolls else None
    pitch_mean = np.mean(avg_pitches) if avg_pitches else None
    pitch_variance = np.var(avg_pitches) if avg_pitches else None

    # Print results
    print("--------------------------------------------------------------")
    print(f"Success rate: {success_count}/{num_experiments}")
    if success_count > 0:
        print(f"Mean traversal time (successful trials): {mean_traversal_time:.2f} seconds")
        print(f"Average roll angle: {roll_mean:.2f} degrees, Variance: {roll_variance:.2f}")
        print(f"Average pitch angle: {pitch_mean:.2f} degrees, Variance: {pitch_variance:.2f}")
    else:
        print("No successful trials")
    print("--------------------------------------------------------------")
    
    return results

def parse_arguments():
    processed_args = []
    for arg in sys.argv[1:]: 
        if '=' in arg and not arg.startswith('-'):
            key, value = arg.split('=', 1)
            processed_args.append(f"--{key}")
            processed_args.append(value)
        else:
            processed_args.append(arg)
    
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run vehicle simulation with various configurations')
    
    # Vehicle and system parameters
    parser.add_argument('--vehicle', type=str, default='hmmwv', help='Vehicle type (default: hmmwv)')
    parser.add_argument('--system', type=str, default='pid', help='Control system type (default: pid)')
    parser.add_argument('--speed', type=float, default=4.0, help='Target vehicle speed (default: 4.0)')
    
    # World parameters
    parser.add_argument('--world_id', type=int, default=1, help='World ID (1-100, default: 1)')
    parser.add_argument('--scale_factor', type=float, default=1.0, 
                        help='Scale factor for terrain (default: 1.0, options: 1.0, 1/6, 1/10)')
    
    # Simulation parameters
    parser.add_argument('--max_time', type=float, default=60.0, help='Maximum simulation time in seconds (default: 60.0)')
    parser.add_argument('--num_experiments', type=int, default=1, 
                        help='Number of experiments to run (default: 1)')
    
    # Visualization parameters
    parser.add_argument('--render', type=lambda x: (str(x).lower() == 'true'), default=True, 
                        help='Enable rendering (default: True)')
    parser.add_argument('--use_gui', type=lambda x: (str(x).lower() == 'true'), default=False, 
                        help='Enable GUI control (default: False)')
    parser.add_argument('--ddv2_sensors', type=lambda x: (str(x).lower() == 'true'), default=False,
                        help='Enable DDV2-aligned L/F/R cameras + lidar (default: False)')
    parser.add_argument('--ddv2_dump_dir', type=str, default='',
                        help='If set with ddv2_sensors, save first few obs npz here')
    parser.add_argument('--ddv2_host', type=str, default='127.0.0.1',
                        help='DDV2 plan server host (system=ddv2)')
    parser.add_argument('--ddv2_port', type=int, default=8765,
                        help='DDV2 plan server port (system=ddv2)')
    parser.add_argument('--ddv2_replan_interval', type=float, default=0.5,
                        help='Seconds between DDV2 replan calls (default: 0.5)')
    parser.add_argument('--ddv2_live_viz', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Matplotlib live panel: cam/BEV/traj/timing (default: True)')
    parser.add_argument('--ddv2_draw_world', type=lambda x: (str(x).lower() == 'true'), default=True,
                        help='Draw predicted waypoints as spheres in Chrono (default: True)')
    
    args = parser.parse_args(processed_args)
    return args

if __name__ == '__main__':
    # Load configuration file
    SetChronoDataDirectories()
    
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create config dictionary from arguments
    config = {
        'vehicle': args.vehicle,
        'speed': args.speed,
        'system': args.system,
        'world_id': args.world_id,
        'max_time': args.max_time,
        'scale_factor': args.scale_factor,
        'render': args.render,
        'use_gui': args.use_gui,
        'ddv2_sensors': args.ddv2_sensors or (args.system == 'ddv2'),
        'ddv2_dump_dir': args.ddv2_dump_dir or None,
        'ddv2_host': args.ddv2_host,
        'ddv2_port': args.ddv2_port,
        'ddv2_replan_interval': args.ddv2_replan_interval,
        'ddv2_live_viz': args.ddv2_live_viz,
        'ddv2_draw_world': args.ddv2_draw_world,
    }
    
    print("--------------------------------------------------------------")
    print("Verti-Bench Configs:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("--------------------------------------------------------------")
    
    # Run simulation
    multiple_experiments(config, num_experiments=args.num_experiments)
    