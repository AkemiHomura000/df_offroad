"""Ensure ROS message packages are importable without a full ROS install.

If ``grid_map_msgs`` (or geometry/nav/std_msgs) is missing, prepend
``verti_bench/third_party/ros_msg_stubs`` to ``sys.path``. Real ROS packages
on PYTHONPATH still take precedence when already importable.
"""

from __future__ import annotations

import sys
from pathlib import Path

# .../verti_bench/envs/utils/ros_msg_stubs.py → parents[2] == verti_bench/
_STUB_ROOT = (
    Path(__file__).resolve().parents[2] / "third_party" / "ros_msg_stubs"
)
_DONE = False


def ensure_ros_msg_stubs(force: bool = False) -> bool:
    """Return True if stubs were added (or already present on path)."""
    global _DONE
    if _DONE and not force:
        return _STUB_ROOT.as_posix() in sys.path

    try:
        import grid_map_msgs  # noqa: F401
        import geometry_msgs  # noqa: F401
        import std_msgs  # noqa: F401
        import nav_msgs  # noqa: F401

        _DONE = True
        return False
    except ImportError:
        pass

    stub = _STUB_ROOT.resolve()
    if not stub.is_dir():
        raise FileNotFoundError(f"ROS msg stubs missing: {stub}")
    path = stub.as_posix()
    if path not in sys.path:
        sys.path.insert(0, path)
    # Smoke-import so failures surface early with a clear path.
    import grid_map_msgs.msg  # noqa: F401
    import geometry_msgs.msg  # noqa: F401
    import std_msgs.msg  # noqa: F401
    import nav_msgs.msg  # noqa: F401

    _DONE = True
    return True
