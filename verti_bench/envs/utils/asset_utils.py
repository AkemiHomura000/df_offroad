import pychrono as chrono
try:
    import pychrono.sensor as sens
except:
    print('Could not import Chrono Sensor')

import random
from PIL import Image, ImageDraw
import numpy as np

class Asset():
    """"Class that initializes an asset"""

    def __init__(self, visual_shape_path, scale=None, collision_shape_path=None, bounding_box=None):
        if (scale == None):
            self.scale = 1
            self.scaled = False
        else:
            self.scale = scale
            self.scaled = False

        self.visual_shape_path = visual_shape_path
        self.collision_shape_path = collision_shape_path

        # Intialize a random material
        self.material = chrono.ChContactMaterialNSC()
        # initialize the body
        self.body = chrono.ChBodyAuxRef()
        # set body as fixed
        self.body.SetFixed(True)

        # Get the visual mesh
        visual_shape_obj = chrono.GetChronoDataFile(visual_shape_path)
        visual_mesh = chrono.ChTriangleMeshConnected().CreateFromWavefrontFile(visual_shape_obj, False, False)
        visual_mesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(scale))
        # Add this mesh to the visual shape
        self.visual_shape = chrono.ChVisualShapeTriangleMesh()
        self.visual_shape.SetMesh(visual_mesh)
        # Add visual shape to the mesh body
        self.body.AddVisualShape(self.visual_shape)

        # Get the collision mesh
        collision_shape_obj = None
        if (collision_shape_path == None):
            # Just use the bounding box
            if (bounding_box == None):
                self.body.EnableCollision(False)
                self.collide_flag = False
            else:
                size = bounding_box * scale
                material = chrono.ChContactMaterialNSC()
                collision_shape = chrono.ChCollisionShapeBox(material, size.x, size.y, size.z)
                self.body.AddCollisionShape(collision_shape)
                self.body.EnableCollision(True)
                self.collide_flag = True
        else:
            collision_shape_obj = chrono.GetChronoDataFile(collision_shape_path)
            collision_mesh = chrono.ChTriangleMeshConnected().CreateFromWavefrontFile(collision_shape_obj, False, False)
            collision_mesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(scale))
            collision_shape = chrono.ChCollisionShapeTriangleMesh(self.material, collision_mesh, 
                                                                  True, True, chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(1))
            self.body.AddCollisionShape(collision_shape)
            # Update the collision model
            self.body.EnableCollision(True)
            self.collide_flag = True

        self.collision_shape = collision_shape_obj
        self.bounding_box = bounding_box

        # Asset's position and orientation will be set by the simulation assets class
        self.pos = chrono.ChVector3d()
        self.ang = 0

    def UpdateAssetPosition(self, pos, ang):
        self.pos = pos
        self.ang = ang
        self.body.SetFrameRefToAbs(chrono.ChFramed(pos, ang))

    # Create a copy constructor for the asset
    def Copy(self):
        """Returns a copy of the asset"""
        asset = Asset(self.visual_shape_path, self.scale, self.collision_shape_path, self.bounding_box)
        return asset

class SimulationAssets():
    """Class that handles assets for the Gym Environment"""

    def __init__(self, system, length, width, SCALE_FACTOR, high_res_data, min_height, max_height, m_isFlat):
        self.system = system
        self.length = length
        self.width = width
        self.SCALE_FACTOR = SCALE_FACTOR
        self.high_res_data = high_res_data
        self.min_height = min_height
        self.max_height = max_height
        self.isFlat = m_isFlat
        self.assets_list = []
        self.positions = []
        
        # Configuration for obstacles placement
        self.VEHICLE_SAFETY_DIST = 12 * self.SCALE_FACTOR
        self.GOAL_SAFETY_DIST = 5 * self.SCALE_FACTOR
        self.ROCK_SAFETY_DIST = 10 * self.SCALE_FACTOR

    def AddAsset(self, asset, number=1):
        """Number of such asset to be added"""
        for _ in range(number):
            new_asset = asset.Copy()
            self.assets_list.append(new_asset)
            
    def get_interpolated_height(self, terrain_array, px_float, py_float, hMin, hMax):
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

    # Position assets relative to goal and chassis
    def RandomlyPositionAssets(self, goal_pos, chassis_body, avoid_positions):
        """Randomly positions assets within the terrain"""
        bmp_dim_y, bmp_dim_x = self.high_res_data.shape
        
        # Calculate transformation factors
        s_norm_x = bmp_dim_x / self.length
        s_norm_y = bmp_dim_y / self.width
        
        T = np.array([
            [s_norm_x, 0, 0],
            [0, s_norm_y, 0],
            [0, 0, 1]
        ])
        
        obstacles_info = {
            'rocks': [],
            'trees': []
        }
        
        # Place assets along the path
        start_pos = chassis_body.GetPos()
        path_vector = goal_pos - start_pos
        path_length = path_vector.Length()
        path_direction = path_vector / path_length
        avoid_positions_ch = [chrono.ChVector3d(pos[0], pos[1], pos[2]) for pos in avoid_positions]
         
        offset_length = self.length
        offset_width = self.width
        
        for asset in self.assets_list:
            placed = False
            attempt_count = 0
            max_attempts = 500
            
            while not placed and attempt_count < max_attempts:
                t = random.uniform(0, path_length) # Random position along the path
                offset_x = random.uniform(-offset_length, offset_length) 
                offset_y = random.uniform(-offset_width, offset_width)
                pos = start_pos + path_direction * t + chrono.ChVector3d(offset_x, offset_y, 0)
                
                # Transform to bitmap coordinates
                pos_chrono = np.array([pos.x + self.length / 2, -pos.y + self.width / 2, 1])
                pos_bmp = np.dot(T, pos_chrono)
                x_bmp = int(np.round(pos_bmp[0]))
                y_bmp = int(np.round(pos_bmp[1]))
                
                # Calculate height from bitmap
                if not (0 <= x_bmp < bmp_dim_x - 1 and 0 <= y_bmp < bmp_dim_y - 1):
                    attempt_count += 1
                    continue
                
                if self.isFlat:
                    pos.z = 0.0
                else:
                    start_height = self.high_res_data[y_bmp, x_bmp]
                    pos.z = self.min_height + start_height
                
                # Check distances
                vehicle_dist = (pos - chassis_body.GetPos()).Length()
                goal_dist = (pos - goal_pos).Length()
                
                # Check if the position is too close to any (start, goal) pair
                close_avoid = any((pos - avoid_pos).Length() < self.VEHICLE_SAFETY_DIST for avoid_pos in avoid_positions_ch)
                
                # Initialize valid position flag
                valid_position = (vehicle_dist > self.VEHICLE_SAFETY_DIST and
                                  goal_dist > self.GOAL_SAFETY_DIST and
                                  not close_avoid)
                    
                # Check distance from other assets
                if valid_position and self.positions:
                    min_pos = min(self.positions, key=lambda x: (x - pos).Length())
                    min_dist = (pos - min_pos).Length()
                    threshold = asset.bounding_box.Length() * asset.scale
                    valid_position = min_dist > max(self.ROCK_SAFETY_DIST, threshold)

                if valid_position:
                    self.positions.append(pos)
                    asset.UpdateAssetPosition(pos, chrono.ChQuaterniond(1, 0, 0, 0))
                    self.system.Add(asset.body)
                    
                    if "rock" in asset.visual_shape_path.lower():
                        obstacles_info['rocks'].append({
                            'position': (pos.x, pos.y, pos.z),
                            'scale': asset.scale
                        })
                    elif "tree" in asset.visual_shape_path.lower():
                        obstacles_info['trees'].append({
                            'position': (pos.x, pos.y, pos.z)
                        })
                        
                    placed = True

                attempt_count += 1
            
            if not placed:
                print(f"Warning: Could not place asset after {max_attempts} attempts")
        
        return obstacles_info

    def CheckContact(self, chassis_body, proper_collision=True):
        """Checks if the chassis is in contact with any asset"""
        # First check if the user wants to check for collision using mesh or bounding box
        if proper_collision:
            # Check for collision using the collision model
            for asset in self.assets_list:
                if (asset.body.GetContactForce().Length() > 0):
                    return 1
            return 0
        else:
            # Check for collision using the absolute position of the asset
            pos = chassis_body.GetPos()
            for asset_pos in self.positions:
                if (pos - asset_pos).Length() < (self.assets_list[0].scale * 2.5):
                    return 1
            return 0
