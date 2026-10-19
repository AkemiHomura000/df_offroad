"""Minimal nuplan stubs so DiffusionDrive model code can import without NAVSIM/nuplan-devkit."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass
from enum import Enum, IntEnum


def install() -> None:
    if "nuplan" in sys.modules and getattr(sys.modules["nuplan"], "_ddv2_stub", False):
        return

    def pkg(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__path__ = []  # type: ignore
        sys.modules[name] = m
        return m

    root = pkg("nuplan")
    root._ddv2_stub = True  # type: ignore

    pkg("nuplan.common")
    pkg("nuplan.common.maps")
    maps = pkg("nuplan.common.maps.abstract_map")

    class SemanticMapLayer(Enum):
        LANE = 1
        INTERSECTION = 2
        WALKWAYS = 3
        LANE_CONNECTOR = 4

    class AbstractMap:
        pass

    class MapObject:
        pass

    maps.SemanticMapLayer = SemanticMapLayer
    maps.AbstractMap = AbstractMap
    maps.MapObject = MapObject

    pkg("nuplan.common.actor_state")
    tracked = pkg("nuplan.common.actor_state.tracked_objects_types")

    class TrackedObjectType(Enum):
        VEHICLE = 1
        PEDESTRIAN = 2
        CZONE_SIGN = 3
        BARRIER = 4
        TRAFFIC_CONE = 5
        GENERIC_OBJECT = 6

    tracked.TrackedObjectType = TrackedObjectType

    state_rep = pkg("nuplan.common.actor_state.state_representation")

    class StateSE2:
        def __init__(self, x=0.0, y=0.0, heading=0.0):
            self.x, self.y, self.heading = x, y, heading

    state_rep.StateSE2 = StateSE2

    oriented = pkg("nuplan.common.actor_state.oriented_box")

    class OrientedBox:
        pass

    oriented.OrientedBox = OrientedBox

    pkg("nuplan.planning")
    pkg("nuplan.planning.simulation")
    pkg("nuplan.planning.simulation.trajectory")
    traj = pkg("nuplan.planning.simulation.trajectory.trajectory_sampling")

    @dataclass
    class TrajectorySampling:
        time_horizon: float = 4.0
        interval_length: float = 0.5

        @property
        def num_poses(self) -> int:
            return int(self.time_horizon / self.interval_length)

    traj.TrajectorySampling = TrajectorySampling

    # BoundingBox2DIndex without importing heavy transfuser_features
    feat = types.ModuleType("navsim.agents.diffusiondrive.transfuser_features")

    class BoundingBox2DIndex(IntEnum):
        _X = 0
        _Y = 1
        _HEADING = 2
        _LENGTH = 3
        _WIDTH = 4

        @classmethod
        def size(cls):
            return 5

        @classmethod
        @property
        def POINT(cls):
            return slice(cls._X, cls._Y + 1)

        @classmethod
        @property
        def HEADING(cls):
            return cls._HEADING

    feat.BoundingBox2DIndex = BoundingBox2DIndex
    sys.modules["navsim.agents.diffusiondrive.transfuser_features"] = feat
