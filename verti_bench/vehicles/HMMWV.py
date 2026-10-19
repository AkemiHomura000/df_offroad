import pychrono as chrono
import pychrono.vehicle as veh
import numpy as np

class HMMWVManager:
    def __init__(self, system, step_size=5e-3):
        self.system = system
        self.step_size = step_size
        self.vehicle = None
        self.chassis_body = None
        self.scale_factor = None
    
    def initialize_vehicle(self, start_pos, goal_pos, terrain_manager):
        """Initialize the HMMWV vehicle with specified parameters"""
        self.vehicle = veh.HMMWV_Reduced(self.system)
        
        # Set vehicle parameters
        self.scale_factor = terrain_manager.scale_factor
        self.vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
        self.vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
        self.vehicle.SetChassisFixed(False)
        self.vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP)  # Higher max torques
        self.vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
        self.vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
        self.vehicle.SetTireType(veh.TireModelType_RIGID)
        self.vehicle.SetTireStepSize(self.step_size)
        self.vehicle.SetInitFwdVel(0.0)
        
        # Initialize position and orientation
        self.init_loc, self.init_rot, self.init_yaw = self.initialize_vw_pos(
            self.vehicle, start_pos, goal_pos, terrain_manager.is_flat, terrain_manager
        )
        
        # Set goal point
        self.goal = self.set_goal(self.system, goal_pos, terrain_manager.is_flat, terrain_manager)
        
        # Initialize the vehicle
        self.vehicle.Initialize()
        
        # Configure differentials
        self.vehicle.LockAxleDifferential(0, True)    
        self.vehicle.LockAxleDifferential(1, True)
        self.vehicle.LockCentralDifferential(0, True)
        self.vehicle.GetVehicle().EnableRealtime(False)
        
        # Set visualization types
        self.vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
        self.vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
        self.vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
        
        # Get chassis body
        self.chassis_body = self.vehicle.GetChassisBody()
        
        return self.vehicle
    
    def initialize_vw_pos(self, vehicle, start_pos, goal_pos, is_flat, terrain_manager):
        """Initialize the vehicle position and orientation"""
        if is_flat:
            start_height = 0
        else:
            # Get height from terrain at start position
            pos_bmp = terrain_manager.transform_to_high_res([start_pos])[0]
            high_res_dim_x = terrain_manager.high_res_dim_x
            high_res_dim_y = terrain_manager.high_res_dim_y
            pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
            pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1))) 
            start_height = terrain_manager.high_res_data[pos_bmp_y, pos_bmp_x]

        # Set position with correct height
        start_pos = (start_pos[0], start_pos[1], start_height * self.scale_factor + start_pos[2])
        
        # Calculate orientation based on direction to goal
        dx = goal_pos[0] - start_pos[0]
        dy = goal_pos[1] - start_pos[1]
        start_yaw = np.arctan2(dy, dx)
        
        # Create location and rotation objects
        init_loc = chrono.ChVector3d(*start_pos)
        init_rot = chrono.QuatFromAngleZ(start_yaw)
        
        # Set position
        vehicle.SetInitPosition(chrono.ChCoordsysd(init_loc, init_rot))
        
        return init_loc, init_rot, start_yaw
        
    def set_goal(self, system, goal_pos, is_flat, terrain_manager):
        """Create a goal marker at the target position"""
        if is_flat:
            goal_height = 0
        else:
            # Get height from terrain at goal position
            pos_bmp = terrain_manager.transform_to_high_res([goal_pos])[0] 
            high_res_dim_x = terrain_manager.high_res_dim_x
            high_res_dim_y = terrain_manager.high_res_dim_y
            pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, high_res_dim_x - 1)))
            pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, high_res_dim_y - 1)))
            goal_height = terrain_manager.high_res_data[pos_bmp_y, pos_bmp_x]
        
        # Set goal position with correct height
        offset = 1.0 * self.scale_factor
        goal_pos = (goal_pos[0], goal_pos[1], goal_height * self.scale_factor + goal_pos[2] + offset)
        goal = chrono.ChVector3d(*goal_pos)

        # Create goal sphere with visualization
        goal_contact_material = chrono.ChContactMaterialNSC()
        goal_body = chrono.ChBodyEasySphere(0.5 * self.scale_factor, 1000, True, False, goal_contact_material)
        goal_body.SetPos(goal)
        goal_body.SetFixed(True)
        
        # Apply red visualization material
        goal_mat = chrono.ChVisualMaterial()
        goal_mat.SetAmbientColor(chrono.ChColor(1, 0, 0)) 
        goal_mat.SetDiffuseColor(chrono.ChColor(1, 0, 0))
        goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
        
        # Add goal to system
        system.Add(goal_body)
        
        return goal
    
    def setup_moving_patches(self, deform_terrains, tracked=False):
        """Add moving patches for deformable terrain under vehicle wheels"""
        if tracked:
            for deform_terrain in deform_terrains:
                deform_terrain.AddMovingPatch(
                    self.chassis_body, 
                    chrono.VNULL, 
                    chrono.ChVector3d(5.0 * self.scale_factor, 3.0 * self.scale_factor, 1.0 * self.scale_factor)
                )
                deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
        
        else:
            for deform_terrain in deform_terrains:
                for axle in self.vehicle.GetVehicle().GetAxles():
                    deform_terrain.AddMovingPatch(
                        axle.m_wheels[0].GetSpindle(), 
                        chrono.VNULL, 
                        chrono.ChVector3d(1.0 * self.scale_factor, 0.6 * self.scale_factor, 1.0 * self.scale_factor)
                    )
                    deform_terrain.AddMovingPatch(
                        axle.m_wheels[1].GetSpindle(), 
                        chrono.VNULL, 
                        chrono.ChVector3d(1.0 * self.scale_factor, 0.6 * self.scale_factor, 1.0 * self.scale_factor)
                    )
                deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    
    def get_position(self):
        """Get current vehicle position"""
        return self.vehicle.GetVehicle().GetPos()
    
    def get_speed(self):
        """Get current vehicle speed"""
        return self.vehicle.GetVehicle().GetSpeed()
    
    def get_rotation(self):
        """Get vehicle rotation in Euler angles"""
        return self.vehicle.GetVehicle().GetRot().GetCardanAnglesXYZ()
        
    def synchronize(self, time, driver_inputs, terrain):
        """Synchronize vehicle with terrain"""
        self.vehicle.Synchronize(time, driver_inputs, terrain)
        
    def advance(self, step_size):
        """Advance vehicle simulation"""
        self.vehicle.Advance(step_size)
        
    def get_chassis_body(self):
        """Get vehicle chassis body"""
        return self.chassis_body
    
    def change_gear(self, gear):
        """Change the vehicle transmission gear"""
        transmission = self.vehicle.GetVehicle().GetTransmission()
        transmission.SetGear(gear)
