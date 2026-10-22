"""Minimal std_msgs stubs for Verti-Bench offline (no ROS)."""


class Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ""


class Float32MultiArray:
    def __init__(self):
        self.data = []
