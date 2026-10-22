"""Minimal nav_msgs stubs for Verti-Bench offline (no ROS)."""

from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Header


class PoseWithCovariance:
    def __init__(self):
        self.pose = Pose()
        self.covariance = [0.0] * 36


class TwistWithCovariance:
    def __init__(self):
        self.twist = Twist()
        self.covariance = [0.0] * 36


class Odometry:
    def __init__(self):
        self.header = Header()
        self.child_frame_id = ""
        self.pose = PoseWithCovariance()
        self.twist = TwistWithCovariance()
