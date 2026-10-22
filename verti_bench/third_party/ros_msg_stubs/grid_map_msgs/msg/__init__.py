"""Minimal grid_map_msgs stubs for Verti-Bench offline (no ROS).

Enough for MPPI/TAL/TNT/WMVCT MapProcessor attribute access; Chrono sims
build maps from Verti terrain rasters rather than live ROS GridMap topics.
"""

from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, Header


class GridMapInfo:
    def __init__(self):
        self.pose = Pose()
        self.length_x = 0.0
        self.length_y = 0.0
        self.resolution = 1.0


class GridMap:
    def __init__(self):
        self.header = Header()
        self.info = GridMapInfo()
        self.layers = []
        self.basic_layers = []
        self.data = []  # list of Float32MultiArray
        self.outer_start_index = 0
