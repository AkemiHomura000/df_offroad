import pychrono as chrono
import pychrono.vehicle as veh
import numpy as np
from PIL import Image
import os
import shutil
import glob
import yaml
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel

import pychrono.sensor as sens
from verti_bench.envs.utils.asset_utils import *

class SCMParameters:
    def __init__(self):
        # Soil Contact Model parameters for deformable terrain
        self.Bekker_Kphi = 0    # Kphi, frictional modulus in Bekker model
        self.Bekker_Kc = 0      # Kc, cohesive modulus in Bekker model
        self.Bekker_n = 0       # n, exponent of sinkage in Bekker model (usually 0.6...1.8)
        self.Mohr_cohesion = 0  # Cohesion in, Pa, for shear failure
        self.Mohr_friction = 0  # Friction angle (in degrees!), for shear failure
        self.Janosi_shear = 0   # J , shear parameter, in meters, in Janosi-Hanamoto formula (usually few mm or cm)
        self.elastic_K = 0      # elastic stiffness K (must be > Kphi very high values gives the original SCM model)
        self.damping_R = 0      # vertical damping R, per unit area (vertical speed proportional, it is zero in original SCM model)

    def SetParameters(self, terrain):
        # Apply parameters to terrain
        terrain.SetSoilParameters(
            self.Bekker_Kphi,    # Bekker Kphi
            self.Bekker_Kc,      # Bekker Kc
            self.Bekker_n,       # Bekker n exponent
            self.Mohr_cohesion,  # Mohr cohesive limit (Pa)
            self.Mohr_friction,  # Mohr friction limit (degrees)
            self.Janosi_shear,   # Janosi shear coefficient (m)
            self.elastic_K,      # Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
            self.damping_R)      # Damping (Pa s/m), proportional to negative vertical speed (optional)

    def InitializeParametersAsSoft(self):
        # Snow parameters
        self.Bekker_Kphi = 0.2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 4e7
        self.damping_R = 3e4
        
    def InitializeParametersAsMid(self):
        # Mud parameters
        self.Bekker_Kphi = 2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 2e8
        self.damping_R = 3e4
        
    def InitializeParametersAsHard(self):
        # Sand parameters
        self.Bekker_Kphi = 5301e3
        self.Bekker_Kc = 102e3
        self.Bekker_n = 0.793
        self.Mohr_cohesion = 1.3e3
        self.Mohr_friction = 31.1
        self.Janosi_shear = 1.2e-2
        self.elastic_K = 4e8
        self.damping_R = 3e4

class TerrainManager:
    def __init__(self, world_id, scale_factor=1.0):
        # Initialize terrain manager
        self.world_id = world_id
        self.config = self._load_config()
        self.scale_factor = scale_factor
        
        # Terrain parameters from config
        self.terrain_length = self.config['terrain']['length'] * scale_factor
        self.terrain_width = self.config['terrain']['width'] * scale_factor
        self.min_terrain_height = self.config['terrain']['min_height'] * scale_factor
        self.max_terrain_height = self.config['terrain']['max_height'] * scale_factor
        self.difficulty = self.config['terrain']['difficulty']
        self.is_flat = self.config['terrain']['is_flat']
        self.positions = self.config['positions']
        self.terrain_type = self.config['terrain_type']
        self.obstacle_flag = self.config['obstacles_flag']
        self.obstacle_density = self.config['obstacle_density']
        self.textures = self.config['textures']
        self.terrain_delta = 0.1 * scale_factor # mesh resolution for SCM terrain
        self.patch_size = 9
        
        # Load terrain data
        self.terrain_path, self.terrain_array, self.bmp_dim_x, self.bmp_dim_y = self._load_terrain_data()
        self.high_res_data, self.high_res_dim_x, self.high_res_dim_y = self._load_high_res_data()
        self.property_dict, self.terrain_labels, self.texture_options, self.terrain_patches = self._load_texture_config()
        self.high_res_terrain_labels = self._load_high_res_terrain_labels()
        self.obs_path = self._load_obstacle_map()
        
    def _load_config(self):
        """Load YAML configuration"""
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "./data/BenchMaps/sampled_maps/Configs/Final", f"config{self.world_id}_*.yaml")
        matched_file = glob.glob(config_path)
        config_path = matched_file[0]
        print(f"Using config file: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
            
        return config
        
    def _load_terrain_data(self):
        """Load main terrain bitmap"""
        terrain_file = f"{self.world_id}.bmp"
        terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "./data/BenchMaps/sampled_maps/Worlds", terrain_file)
        
        terrain_image = Image.open(terrain_path)
        terrain_array = np.array(terrain_image)
        bmp_dim_y, bmp_dim_x = terrain_array.shape 
        if (bmp_dim_y, bmp_dim_x) != (129, 129):
            raise ValueError("Check terrain file and dimensions")

        return terrain_path, terrain_array, bmp_dim_x, bmp_dim_y
    
    def _load_obstacle_map(self):
        """Load obstacle map"""
        obs_file = f"obs{self.world_id}_{self.difficulty}.bmp"
        obs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "./data/BenchMaps/sampled_maps/Configs/Final", obs_file)
        return obs_path
        
    def _load_high_res_data(self):
        """Load high resolution terrain data"""
        high_res_file = f"height{self.world_id}_*.npy"
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
        
        return high_res_data, high_res_dim_x, high_res_dim_y
    
    def _load_high_res_terrain_labels(self):
        """Load high resolution terrain labels"""
        high_res_factor = self.high_res_dim_x // self.terrain_labels.shape[1]
        high_res_terrain_labels = np.zeros((self.terrain_labels.shape[0] * high_res_factor, 
                                            self.terrain_labels.shape[1] * high_res_factor), dtype=np.int32)
        for i in range(self.terrain_labels.shape[0]):
            for j in range(self.terrain_labels.shape[1]):
                label_value = self.terrain_labels[i, j]
                i_start = i * high_res_factor
                j_start = j * high_res_factor
                high_res_terrain_labels[i_start:i_start+high_res_factor, 
                                        j_start:j_start+high_res_factor] = label_value

        return high_res_terrain_labels
        
    def _load_texture_config(self):
        """Setup texture configurations"""
        property_dict = {}
        terrain_patches = []
        
        labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "./data/BenchMaps/sampled_maps/Configs/Final", f"labels{self.world_id}_*.npy")
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
        for texture_info in self.textures:
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
        
    def terrain_patch_bmp(self, terrain_array, start_y, end_y, start_x, end_x, idx):
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
        
    def deformable_params(self, terrain_type):
        """Initialize SCM parameters based on terrain type"""
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
        
    def transform_to_bmp(self, chrono_positions):
        """Transform PyChrono coordinates to bitmap coordinates"""
        bmp_dim_y, bmp_dim_x = self.terrain_array.shape
    
        # Normalization factors
        s_norm_x = bmp_dim_x / (2 * self.terrain_length)
        s_norm_y = bmp_dim_y / (2 * self.terrain_width)
        
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
            pos_chrono = np.array([vehicle_x + self.terrain_length, vehicle_y + self.terrain_width, 1])
            
            # Transform to BMP coordinates
            pos_bmp = np.dot(T, pos_chrono)
            bmp_positions.append((pos_bmp[0], pos_bmp[1]))
        
        return bmp_positions
        
    def transform_to_chrono(self, bmp_positions):
        """Transform bitmap coordinates to PyChrono coordinates"""
        bmp_dim_y, bmp_dim_x = self.terrain_array.shape  
    
        # Inverse normalization factors
        s_norm_x = bmp_dim_x / (2 * self.terrain_length)
        s_norm_y = bmp_dim_y / (2 * self.terrain_width)

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
            x = (pos_chrono[0] - self.terrain_length) 
            y = -(pos_chrono[1] - self.terrain_width) 
            chrono_positions.append((x, y))

        return chrono_positions
        
    def transform_to_high_res(self, chrono_positions, height_array=None):
        """Transform PyChrono coordinates to high-res bitmap coordinates"""
        if height_array is None:
            height_array = self.high_res_data
            
        bmp_dim_y, bmp_dim_x = height_array.shape
        
        # Normalization factors
        s_norm_x = bmp_dim_x / (2 * self.terrain_length)
        s_norm_y = bmp_dim_y / (2 * self.terrain_width)
        
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
            pos_chrono = np.array([vehicle_x + self.terrain_length, vehicle_y + self.terrain_width, 1])
            
            # Transform to BMP coordinates
            pos_bmp = np.dot(T, pos_chrono)
            bmp_positions.append((pos_bmp[0], pos_bmp[1]))
        
        return bmp_positions
    
    def transform_to_high_res_torch(self, chrono_positions, height_array=None, device='cuda'):
        # If height array is not provided, use the default one
        if height_array is None:
            bmp_dim_y, bmp_dim_x = self.high_res_data.shape
        else:
            bmp_dim_y, bmp_dim_x = height_array.shape
            
        # Normalization factors
        s_norm_x = bmp_dim_x / (2 * self.terrain_length)
        s_norm_y = bmp_dim_y / (2 * self.terrain_width)
        
        # Transformation matrix
        T = torch.tensor([
            [s_norm_x, 0, 0],
            [0, s_norm_y, 0],
            [0, 0, 1]
        ], dtype=torch.float32, device=device)
        
        vehicle_x = chrono_positions[:, 0]
        vehicle_y = -chrono_positions[:, 1]
        
        # Create the homogeneous coordinates
        pos_chrono = torch.stack(
            (vehicle_x + self.terrain_length, 
            vehicle_y + self.terrain_width, 
            torch.ones_like(vehicle_x, device=device)), 
            dim=1
        )
        
        # Transform to BMP coordinates using batch matrix multiplication
        pos_bmp = torch.matmul(pos_chrono, T.t())  # Transpose T to align dimensions
        
        # Return the transformed positions (x, y only)
        return pos_bmp[:, :2]
    
    def find_regular_shape(self, patch_size, max_dim):
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
    
    def best_shape_fit(self, shapes, patch_size, available_patches):
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
            
    def combine_rigid(self, system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size):
        """Combine patches into larger sections for rigid terrain"""
        rigid_sections = []
        max_dim = terrain_labels.shape[0]
        
        rigid_patches = defaultdict(set)
        for patch_file, i, j, center_pos in terrain_patches:
            label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
            if not texture_options[label]['is_deformable']:
                rigid_patches[label].add((i, j, center_pos))
                
        processed_patches = set()
        shapes = self.find_regular_shape(patch_size, max_dim)
        
        for label, patches in rigid_patches.items():
            patch_coords = {(i, j) for i, j, _ in patches}
            
            while patch_coords:
                best_shape, selected_patches = self.best_shape_fit(shapes, patch_size, patch_coords)
                
                if not best_shape or not selected_patches:
                    break
                
                width, height = best_shape
                patches_width = (width - 1) // (patch_size - 1) + 1
                patches_height = (height - 1) // (patch_size - 1) + 1
                width_scaled = width * self.scale_factor
                height_scaled = height * self.scale_factor
                
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
                section_pos = chrono.ChVector3d(avg_x * self.scale_factor, avg_y * self.scale_factor, 0)
                
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
                rigid_terrain = veh.RigidTerrain(system)
                patch_mat = chrono.ChContactMaterialNSC()
                patch_mat.SetFriction(properties['friction'])
                patch_mat.SetRestitution(properties['restitution'])
                
                # Apply scaling
                if self.is_flat:
                    patch = rigid_terrain.AddPatch(patch_mat, chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot), 
                                                   width_scaled - self.scale_factor, height_scaled - self.scale_factor)
                else:
                    start_i = min_i * (patch_size - 1)
                    end_i = max_i * (patch_size - 1) + patch_size
                    start_j = min_j * (patch_size - 1)
                    end_j = max_j * (patch_size - 1) + patch_size
                    
                    file = self.terrain_patch_bmp(self.terrain_array,
                                                start_i, end_i,
                                                start_j, end_j,
                                                len(rigid_sections))
                                        
                    patch = rigid_terrain.AddPatch(patch_mat,
                                                chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot),
                                                file,
                                                width_scaled - self.scale_factor, height_scaled - self.scale_factor,
                                                self.min_terrain_height,
                                                self.max_terrain_height)
                
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
                patch_pos = chrono.ChVector3d(*center_pos) * self.scale_factor
                
                rigid_terrain = veh.RigidTerrain(system)
                patch_mat = chrono.ChContactMaterialNSC()
                patch_mat.SetFriction(properties['friction'])
                patch_mat.SetRestitution(properties['restitution'])
                
                scaled_patch_size = patch_size * self.scale_factor
                if self.is_flat:
                    patch = rigid_terrain.AddPatch(patch_mat,
                                                chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                                scaled_patch_size - self.scale_factor, scaled_patch_size - self.scale_factor)
                else:
                    patch = rigid_terrain.AddPatch(patch_mat,
                                                chrono.ChCoordsysd(patch_pos, chrono.CSYSNORM.rot),
                                                patch_file,
                                                scaled_patch_size - self.scale_factor, scaled_patch_size - self.scale_factor,
                                                self.min_terrain_height,
                                                self.max_terrain_height)
                                                
                patch.SetTexture(veh.GetDataFile(properties['texture_file']), patch_size, patch_size)
                rigid_terrain.Initialize()
                rigid_sections.append(rigid_terrain)
        
        return rigid_sections, property_dict, terrain_labels
        
    def combine_deformation(self, system, terrain_patches, property_dict, texture_options):
        """Set up deformable terrain sections"""
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
        bmp_width, bmp_height = self.terrain_array.shape
        
        if num_textures == 1:
            terrain_type = terrain_types[0]
            center_x, center_y = bmp_width // 2, bmp_height // 2
            chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * self.scale_factor, 
                                            chrono_center_y - 0.5 * self.scale_factor, 0)
                
            # Create terrain section
            deform_terrain = veh.SCMTerrain(system)
            
            # Set SCM parameters
            terrain_params = self.deformable_params(terrain_type)
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
            width = 2 * self.terrain_length - 1 * self.scale_factor
            height = 2 * self.terrain_width - 1 * self.scale_factor
            
            # Create and set boundary
            aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
            deform_terrain.SetBoundary(aabb)  
            
            if self.is_flat:
                deform_terrain.Initialize(width, height, self.terrain_delta)
            else:
                deform_terrain.Initialize(
                    self.terrain_path,
                    width,
                    height,
                    self.min_terrain_height,
                    self.max_terrain_height,
                    self.terrain_delta
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
                chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
                section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * self.scale_factor, 
                                                chrono_center_y - 0.5 * self.scale_factor, 0)
                
                # Create terrain section
                deform_terrain = veh.SCMTerrain(system)
                
                # Set SCM parameters
                terrain_params = self.deformable_params(terrain_type)
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
                width = 2 * self.terrain_length - 1 * self.scale_factor
                height = (section_height - 1) * (2 * self.terrain_width / bmp_height)
                
                # Create and set boundary
                aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
                deform_terrain.SetBoundary(aabb)  
                
                if self.is_flat:
                    deform_terrain.Initialize(
                        width,
                        height,
                        self.terrain_delta
                    )
                else:
                    file = self.terrain_patch_bmp(self.terrain_array, start_y, end_y, 0, bmp_width, idx)
                    deform_terrain.Initialize(
                        file,
                        width,
                        height,
                        self.min_terrain_height,
                        self.max_terrain_height,
                        self.terrain_delta
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
                    
                chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
                section_pos = chrono.ChVector3d(chrono_center_x + 0.5 * self.scale_factor, 
                                                chrono_center_y - 1 * self.scale_factor, 0)
                
                # Create terrain section
                deform_terrain = veh.SCMTerrain(system)
                
                # Set SCM parameters
                terrain_params = self.deformable_params(terrain_type)
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
                
                width = 2 * self.terrain_length - 1 * self.scale_factor
                height = (section_height - 1) * (2 * self.terrain_width / bmp_height)

                # Create and set boundary
                aabb = chrono.ChAABB(chrono.ChVector3d(-width/2, -height/2, 0), chrono.ChVector3d(width/2, height/2, 0))
                deform_terrain.SetBoundary(aabb)
                
                if self.is_flat:
                    deform_terrain.Initialize(width, height, self.terrain_delta)
                else:
                    file = self.terrain_patch_bmp(self.terrain_array, start_y, end_y, 0, bmp_width, idx)
                    deform_terrain.Initialize(
                        file,
                        width,
                        height,
                        self.min_terrain_height,
                        self.max_terrain_height,
                        self.terrain_delta
                    )
                    
                label = type_to_label[terrain_type]
                texture_file = texture_options[label]['texture_file']
                deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
                deform_terrains.append(deform_terrain)
                
        return deform_terrains
        
    def mixed_terrain(self, system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size):
        """Set up mixed terrain with both rigid and deformable sections"""
        deformable_sections = []
        max_dim = terrain_labels.shape[0]
        
        deformable_patches = defaultdict(set)
        for patch_file, i, j, center_pos in terrain_patches:
            label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
            if texture_options[label]['is_deformable']:
                deformable_patches[label].add((i, j, center_pos))
                
        processed_patches = set()
        shapes = self.find_regular_shape(patch_size, max_dim)
        
        for label, patches in deformable_patches.items():
            patch_coords = {(i, j) for i, j, _ in patches}
            best_shape, selected_patches = self.best_shape_fit(shapes, patch_size, patch_coords)
            
            if not best_shape or not selected_patches:
                continue

            width, height = best_shape
            patches_width = (width - 1) // (patch_size - 1)
            patches_height = (height - 1) // (patch_size - 1)
            
            # Create deformable terrain for this shape
            deform_terrain = veh.SCMTerrain(system)
            terrain_type = texture_options[label]['terrain_type']
            terrain_params = self.deformable_params(terrain_type)
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
            section_pos = chrono.ChVector3d(avg_x * self.scale_factor, avg_y * self.scale_factor, 0)
            
            # Initialize terrain section
            deform_terrain.SetPlane(chrono.ChCoordsysd(section_pos, chrono.CSYSNORM.rot))
            deform_terrain.SetMeshWireframe(False)
            
            width_scaled = width * self.scale_factor
            height_scaled = height * self.scale_factor
            # Create and set boundary
            aabb = chrono.ChAABB(chrono.ChVector3d(-width_scaled/2, -height_scaled/2, 0), chrono.ChVector3d(width_scaled/2, height_scaled/2, 0))
            deform_terrain.SetBoundary(aabb)  
            
            if self.is_flat:
                deform_terrain.Initialize(width_scaled - self.scale_factor, height_scaled - self.scale_factor, self.terrain_delta)
            else:
                start_i = min_i * (patch_size - 1)
                end_i = max_i * (patch_size - 1) + patch_size
                start_j = min_j * (patch_size - 1)
                end_j = max_j * (patch_size - 1) + patch_size
                file = self.terrain_patch_bmp(self.terrain_array, 
                                            start_i, end_i,
                                            start_j, end_j,
                                            len(deformable_sections))
                deform_terrain.Initialize(
                    file,
                    width_scaled - self.scale_factor, height_scaled - self.scale_factor,
                    self.min_terrain_height,
                    self.max_terrain_height,
                    self.terrain_delta
                )
            
            # Set texture
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), patches_width, patches_height)
            deformable_sections.append(deform_terrain)
            processed_patches.update(selected_patches)
                
        # Convert remaining deformable patches to first rigid texture
        first_rigid_label = min(label for label, info in texture_options.items() if not info['is_deformable'])
        first_rigid_info = next(info for info in self.textures if info['terrain_type'] == texture_options[first_rigid_label]['terrain_type'])
        
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
    
    def get_cropped_map(self, vehicle, vehicle_pos, region_size, num_front_regions):
        """Get terrain height maps around the vehicle"""
        bmp_dim_y, bmp_dim_x = self.high_res_data.shape  # height (rows), width (columns)
        pos_bmp = self.transform_to_high_res([vehicle_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, self.high_res_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, self.high_res_dim_y - 1)))
        # Check if pos_bmp_x and pos_bmp_y are within bounds
        assert 0 <= pos_bmp_x < self.high_res_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
        assert 0 <= pos_bmp_y < self.high_res_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

        center_x = bmp_dim_x // 2
        center_y = bmp_dim_y // 2
        shift_x = center_x - pos_bmp_x
        shift_y = center_y - pos_bmp_y

        # Shift the map to center the vehicle position
        shifted_map = np.roll(self.high_res_data, shift_y, axis=0)  # y shift affects rows (axis 0)
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
    
    def get_cropped_map_torch(self, vehicle_pos_batch, region_size, batch_size, device='cuda'):
        bmp_dim_y, bmp_dim_x = self.high_res_data.shape
        vehicle_heading_batch = vehicle_pos_batch[:, 5]

        # Convert vehicle positions to BMP coordinates
        pos_bmp = self.transform_to_high_res_torch(vehicle_pos_batch[:, :2], device=device)
        pos_bmp_x = torch.round(torch.clamp(pos_bmp[:, 0], 0, self.high_res_dim_x - 1)).long()
        pos_bmp_y = torch.round(torch.clamp(pos_bmp[:, 1], 0, self.high_res_dim_y - 1)).long()

        # Shifting
        center_x = self.high_res_dim_x // 2
        center_y = self.high_res_dim_y // 2
        shift_x = center_x - pos_bmp_x
        shift_y = center_y - pos_bmp_y

        # Full terrain tensor
        high_res_data_copy = np.array(self.high_res_data, copy=True)
        terrain_tensor = torch.tensor(high_res_data_copy, device=device, dtype=torch.float32)

        half_size = region_size // 2
        result_list = []

        for i in range(batch_size):
            # Shift terrain
            grid_y = torch.arange(bmp_dim_y, device=device).view(-1, 1)
            grid_x = torch.arange(bmp_dim_x, device=device).view(1, -1)
            shifted_y = (grid_y + shift_y[i]) % bmp_dim_y
            shifted_x = (grid_x + shift_x[i]) % bmp_dim_x
            shifted_map = terrain_tensor[shifted_y, shifted_x]

            # Rotate
            angle_degrees = (torch.rad2deg(vehicle_heading_batch[i]) % 360).item()
            map_tensor = shifted_map.unsqueeze(0).unsqueeze(0)
            rotated_map = F.rotate(map_tensor, -angle_degrees)
            rotated_map = rotated_map.squeeze(0).squeeze(0)
            rotated_map = torch.fliplr(rotated_map)

            # Crop with bounds
            center_y_pos, center_x_pos = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
            start_y = center_y_pos - half_size
            end_y = center_y_pos + half_size
            start_x = center_x_pos - half_size
            end_x = center_x_pos + half_size

            # Clamp crop indices
            start_y = max(start_y, 0)
            end_y = min(end_y, rotated_map.shape[0])
            start_x = max(start_x, 0)
            end_x = min(end_x, rotated_map.shape[1])

            cropped = rotated_map[start_y:end_y, start_x:end_x]
            result_list.append(cropped.T)

        return torch.stack(result_list)

    
    def get_current_label(self, vehicle, vehicle_pos, region_size):
        """Get terrain type labels beneath the vehicle"""
        bmp_dim_y, bmp_dim_x = self.high_res_terrain_labels.shape  # height (rows), width (columns)
        pos_bmp = self.transform_to_high_res([vehicle_pos], self.high_res_terrain_labels)[0]
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
        shifted_labels = np.roll(self.high_res_terrain_labels, shift_y, axis=0) 
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
    
    def find_similar_region(self, under_vehicle, front_regions, region_size,
                            vehicle_global_pos, vehicle_global_heading):
        """
        Identify the terrain region in front of the vehicle that has the most 
        similar characteristics to the terrain currently underneath the vehicle.
        """
        under_mean = np.mean(under_vehicle)
        under_var = np.var(under_vehicle)

        # Calculate mean and variance for each front region
        region_stats = []
        for idx, region in enumerate(front_regions):
            region_mean = np.mean(region)
            region_var = np.var(region)
            region_stats.append({
                'index': idx, 'mean': region_mean, 'variance': region_var,
                'distance': abs(region_mean - under_mean)
            })

        # Primary criterion: mean similarity (distance to current mean)
        # Secondary criterion: variance (lower is better)
        sorted_regions = sorted(
            region_stats,
            key=lambda stat: (stat['distance'], stat['variance'])
        )
        best_region = sorted_regions[0]
        best_index = best_region['index']
        num_front_regions = len(front_regions)
        center_index = num_front_regions // 2
        
        map_cols = (2 * self.terrain_length) / self.high_res_dim_x
        map_rows = (2 * self.terrain_width) / self.high_res_dim_y
        local_chrono_x_meters = region_size * map_rows
        local_chrono_y_meters = -((best_index - center_index) * region_size) * map_cols
        gx = vehicle_global_pos[0]
        gy = vehicle_global_pos[1]
        yaw = vehicle_global_heading

        target_global_x = gx + local_chrono_x_meters * np.cos(yaw) - \
                               local_chrono_y_meters * np.sin(yaw)
        target_global_y = gy + local_chrono_x_meters * np.sin(yaw) + \
                               local_chrono_y_meters * np.cos(yaw)

        return target_global_x, target_global_y
    
    def add_obstacles(self, system):
        """Add rocks and trees to the terrain"""
        m_assets = SimulationAssets(system, self.terrain_length * 1.8, self.terrain_width * 1.8, self.scale_factor,
                                self.high_res_data, self.min_terrain_height, self.max_terrain_height, self.is_flat)
    
        # Add rocks
        for rock_info in self.config['obstacles']['rocks']:
            rock_scale = rock_info['scale'] * self.scale_factor
            rock_pos = chrono.ChVector3d(rock_info['position']['x'] * self.scale_factor,
                                    rock_info['position']['y'] * self.scale_factor,
                                    rock_info['position']['z'] * self.scale_factor)
            
            rock = Asset(visual_shape_path="sensor/offroad/rock.obj",
                        scale=rock_scale,
                        bounding_box=chrono.ChVector3d(4.4, 4.4, 3.8))
            
            asset_body = rock.Copy()
            asset_body.UpdateAssetPosition(rock_pos, chrono.ChQuaterniond(1, 0, 0, 0))
            system.Add(asset_body.body)
        
        # Add trees
        for tree_info in self.config['obstacles']['trees']:
            tree_pos = chrono.ChVector3d(tree_info['position']['x'] * self.scale_factor,
                                    tree_info['position']['y'] * self.scale_factor,
                                    tree_info['position']['z'] * self.scale_factor)
            
            tree = Asset(visual_shape_path="sensor/offroad/tree.obj",
                        scale=1.0 * self.scale_factor,
                        bounding_box=chrono.ChVector3d(1.0, 1.0, 5.0))
            
            asset_body = tree.Copy()
            asset_body.UpdateAssetPosition(tree_pos, chrono.ChQuaterniond(1, 0, 0, 0))
            system.Add(asset_body.body)
        
        return m_assets
        
    def generate_obstacle_map(self):
        """Generate bitmap marking obstacles for path planning"""
        obs_terrain = self.terrain_array.copy()
        
        # Process rocks and trees from config
        for rock in self.config['obstacles']['rocks']:
            # Get rock position and scale
            rock_pos = rock['position'] 
            rock_scale = rock['scale']
            
            # Create ChVector3d for position
            obstacle_pos = chrono.ChVector3d(rock_pos['x'] * self.scale_factor, rock_pos['y'] * self.scale_factor, rock_pos['z'] * self.scale_factor)
            
            # Transform obstacle position to bitmap coordinates
            obstacle_bmp = self.transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
            obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, self.bmp_dim_x - 1)))
            obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, self.bmp_dim_y - 1)))
            
            # Rock bounding box is 4.4 x 4.4 x 3.8
            box_width = 4.4 * rock_scale * self.scale_factor
            box_length = 4.4 * rock_scale * self.scale_factor
                
            # Create a mask for the obstacle
            width_pixels = int(box_width * self.bmp_dim_x / (2 * self.terrain_length))
            length_pixels = int(box_length * self.bmp_dim_y / (2 * self.terrain_width))
            
            # Calculate bounds for the obstacle footprint
            x_min = max(0, obs_x - width_pixels // 2)
            x_max = min(self.bmp_dim_x, obs_x + width_pixels // 2 + 1)
            y_min = max(0, obs_y - length_pixels // 2)
            y_max = min(self.bmp_dim_y, obs_y + length_pixels // 2 + 1)
            
            obs_terrain[y_min:y_max, x_min:x_max] = 255
            
        for tree in self.config['obstacles']['trees']:
            # Get tree position
            tree_pos = tree['position'] 
            
            # Create ChVector3d for position
            obstacle_pos = chrono.ChVector3d(tree_pos['x'] * self.scale_factor, tree_pos['y'] * self.scale_factor, tree_pos['z'] * self.scale_factor)
            
            # Transform obstacle position to bitmap coordinates
            obstacle_bmp = self.transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
            obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, self.bmp_dim_x - 1)))
            obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, self.bmp_dim_y - 1)))
            
            # Tree bounding box is 1.0 x 1.0 x 5.0
            box_width = 1.0 * self.scale_factor
            box_length = 1.0 * self.scale_factor
                
            # Create a mask for the obstacle
            width_pixels = int(box_width * self.bmp_dim_x / (2 * self.terrain_length))
            length_pixels = int(box_length * self.bmp_dim_y / (2 * self.terrain_width))
            
            # Calculate bounds for the obstacle footprint
            x_min = max(0, obs_x - width_pixels // 2)
            x_max = min(self.bmp_dim_x, obs_x + width_pixels // 2 + 1)
            y_min = max(0, obs_y - length_pixels // 2)
            y_max = min(self.bmp_dim_y, obs_y + length_pixels // 2 + 1)
            
            obs_terrain[y_min:y_max, x_min:x_max] = 255
            
        # Save obstacle map
        obs_terrain_image = Image.fromarray(obs_terrain.astype(np.uint8), mode='L')
        obs_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                        "./data/BenchMaps/sampled_maps/Configs/Custom", f"obs{self.world_id}_{self.difficulty}.bmp")
        os.makedirs(os.path.dirname(obs_path), exist_ok=True)
        obs_terrain_image.save(obs_path)
        
        return obs_path
        
    def initialize_terrain(self, system):
        """Initialize terrain in the system based on configuration"""
        terrain_objects = []
        
        if self.terrain_type == 'rigid':
            # Initialize rigid terrain
            original_labels = self.terrain_labels.copy()
            rigid_terrains, _, _ = self.combine_rigid(
                system, self.terrain_patches, self.terrain_labels.copy(),
                self.property_dict, self.texture_options, self.patch_size
            )
            terrain_objects.extend(rigid_terrains)
            self.terrain_labels = original_labels
            
        elif self.terrain_type == 'deformable':
            # Initialize deformable terrain
            original_labels = self.terrain_labels.copy()
            deform_terrains = self.combine_deformation(
                system, self.terrain_patches, self.property_dict, self.texture_options
            )
            terrain_objects.extend(deform_terrains)
            self.terrain_labels = original_labels
        
        else: 
            # Initialize mixed terrain
            original_labels = self.terrain_labels.copy()
            deform_terrains, _, _ = self.mixed_terrain(
                system, self.terrain_patches, self.terrain_labels.copy(), self.property_dict,
                self.texture_options, self.patch_size
            )
            rigid_terrains, _, _ = self.combine_rigid(
                system, self.terrain_patches, original_labels, self.property_dict,
                self.texture_options, self.patch_size
            )
            terrain_objects.extend(deform_terrains)  
            terrain_objects.extend(rigid_terrains)  
            self.terrain_labels = original_labels
            
        # Add obstacles if needed
        if self.obstacle_flag:
            self.add_obstacles(system)
            
        return terrain_objects
    