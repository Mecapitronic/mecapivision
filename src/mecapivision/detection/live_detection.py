from typing import Sequence

import click
import cv2 as cv
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike
from loguru import logger

from .._utils import (
    CANT_RECEIVE_FRAME,
    get_last_camera,
    load_camera_calibration,
)


# command line entry with lick that accept the name of a file where the camera calibration is stored
@click.command()
@click.option(
    "--calibration_file",
    "-c",
    help="File where the camera calibration is stored",
)
def cli(calibration_file: str) -> None:
    detect_aruco_live(get_last_camera(), calibration_file)


def detect_aruco_live(
    video: str,
    calibration_file: str,
) -> None:
    logger.info("Live detection. Press 'q' to quit")

    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    detector = get_detector()
    camera_matrix, dist_coeffs = load_camera_calibration(calibration_file)

    while camera.isOpened():
        ret, image = camera.read()

        if not ret:
            print(CANT_RECEIVE_FRAME)
            break

        marker_corners, marker_ids, rejected_candidates = detect_aruco(image, detector)
        if marker_corners:
            estimate_pose_aruco(
                image,
                camera_matrix,
                dist_coeffs,
                marker_corners,
                marker_ids,
                rejected_candidates,
            )

        cv.imshow("Live Cam", image)

        if cv.waitKey(5) & 0xFF == ord("q"):
            logger.info("Quitting")
            break

    camera.release()
    cv.destroyAllWindows()


def get_detector() -> cv.aruco.ArucoDetector:
    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    return detector


def detect_aruco(image: MatLike, detector: cv.aruco.ArucoDetector) -> tuple:
    result: tuple[
        Sequence[MatLike],
        MatLike,
        Sequence[MatLike],
    ] = detector.detectMarkers(image)

    (marker_corners, marker_ids, rejected_candidates) = result

    # we should output_image = input_image.copy()
    # but we live stream so we don't need to preserve original image
    aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

    logger.debug("results:\n")
    logger.debug(f"marker corners: {marker_corners}")
    logger.debug(f"marker ids: {marker_ids}")
    logger.debug(f"rejected candidates: {rejected_candidates}")

    # might be useless as we are drawing on the image in this function
    return marker_corners, marker_ids, rejected_candidates


def estimate_pose_aruco(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    marker_corners: Sequence[MatLike],
    marker_ids: MatLike,
    rejected_candidates: Sequence[MatLike],
    *,
    show_rejected: bool = True,
) -> None:
    print("searching position of aruco markers")

    rvecs = []
    tvecs = []

    # Set coordinate system; we are on a plane surface
    obj_points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )

    tick = cv.getTickCount()

    n_markers: int = len(marker_corners)
    logger.debug(f"detected {n_markers} markers")
    n_ids: int = len(marker_ids)
    logger.debug(f"detected ids: {marker_ids}")

    # Calculate pose for each marker
    for i in range(n_markers):
        logger.debug(f"calculating pose for marker {marker_ids[i]}")
        logger.debug(f"marker corners: {marker_corners[i][0]}")
        logger.debug(f"obj points: {obj_points}")

        ret, rvec, tvec = cv.solvePnP(
            obj_points,
            marker_corners[i],
            camera_matrix,
            dist_coeffs,
        )
        if not ret:
            print(f"Pose estimation failed for marker {marker_ids[i]}")

        rvecs.append(rvec)
        tvecs.append(tvec)

    current_time = (cv.getTickCount() - tick) / cv.getTickFrequency()
    print(f"Detection Time = {current_time * 1000} ms")

    # draw results
    if n_ids > 0:
        # already done in detection section
        # cv.aruco.drawDetectedMarkers(image, marker_corners, marker_ids)

        for i in range(n_ids):
            cv.drawFrameAxes(
                image,
                camera_matrix,
                dist_coeffs,
                rvecs[i],
                tvecs[i],
                n_markers * 1.5,
                2,
            )

        if show_rejected and len(rejected_candidates):
            cv.aruco.drawDetectedMarkers(image, rejected_candidates, None)
