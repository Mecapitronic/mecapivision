"""Calibration module for camera calibration

see https://medium.com/@amit25173/opencv-camera-calibration-03d19f0f52bc for more
"""

from .chessboard import calibrate_camera_with_chessboard, calibrate_fake_camera
from .record import record_pictures_cli

__all__ = [
    "calibrate_fake_camera",
    "calibrate_camera_with_chessboard",
    "record_pictures_cli",
]
