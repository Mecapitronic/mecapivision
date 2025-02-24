"""Calibration module for camera calibration

see https://medium.com/@amit25173/opencv-camera-calibration-03d19f0f52bc for more
"""

from .chessboard import calibrate_camera_with_chessboard
from .fake_chessboard import calibrate_fake_camera

__all__ = [
    "calibrate_fake_camera",
    "calibrate_camera_with_chessboard",
]
