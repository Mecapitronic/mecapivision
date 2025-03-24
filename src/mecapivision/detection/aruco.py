from typing import Any, Sequence

import cv2 as cv
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike
from loguru import logger

from .._utils import read_parameters


def get_aruco_tag(aruco_id: int, size_in_pixels: int = 200) -> MatLike:
    logger.info(f"generating aruco tag for id {aruco_id}")
    all_aruco_wards: aruco.Dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    marker_image: MatLike = aruco.generateImageMarker(
        all_aruco_wards, aruco_id, size_in_pixels
    )
    return marker_image


def detect_aruco(image_path: str, show_img: bool = False) -> tuple:
    logger.info(f"searching for aruco markers in image {image_path}")
    input_image: MatLike = cv.imread(image_path, cv.IMREAD_COLOR)

    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    result: tuple[
        Sequence[MatLike],
        MatLike,
        Sequence[MatLike],
    ] = detector.detectMarkers(input_image)

    (marker_corners, marker_ids, rejected_candidates) = result

    logger.debug(
        f"results:\n marker corners: {marker_corners} \n marker ids: {marker_ids} \n Rejected Candidates: {rejected_candidates}"
    )

    output_image: MatLike = input_image.copy()
    aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids)

    if show_img:
        cv.imshow("output", output_image)

    return marker_corners, marker_ids, rejected_candidates


def get_arucos_positions(corners) -> list:
    positions = []
    for corner in corners:
        # Calculate the center of the marker
        center = np.mean(corner[0], axis=0)
        positions.append(center)

    logger.debug(positions)
    return positions


def estimate_pose_aruco(
    image_path: str,
    estimate_pose: bool = True,
    show_rejected: bool = True,
) -> None:
    camera_matrix, dist_coeffs = read_parameters()
    rvecs = []
    tvecs = []

    # Set coordinate system; we are on a plane surface
    obj_points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )

    tick = cv.getTickCount()

    # detect markers
    logger.info(f"searching for aruco markers in image {image_path}")
    input_image: MatLike = cv.imread(image_path, cv.IMREAD_COLOR)
    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)
    marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(
        input_image
    )

    n_markers: int = len(marker_corners)
    n_ids: int = len(marker_ids)

    logger.debug(f"detected {n_markers} markers \n detected ids: {marker_ids}")

    if estimate_pose:
        # Calculate pose for each marker
        for i in range(n_markers):
            logger.info(f"calculating pose for marker {marker_ids[i]}")
            logger.debug(
                f"marker corners: {marker_corners[i][0]} \n obj points: {obj_points}"
            )

            ret, rvec, tvec = cv.solvePnP(
                obj_points,
                marker_corners[i],
                camera_matrix,
                dist_coeffs,
            )
            if not ret:
                logger.warning(f"Pose estimation failed for marker {marker_ids[i]}")

            rvecs.append(rvec)
            tvecs.append(tvec)

        current_time = (cv.getTickCount() - tick) / cv.getTickFrequency()
        logger.info(f"Detection Time = {current_time * 1000} ms")

    # draw results
    input_image = cv.imread(image_path, cv.IMREAD_COLOR)
    output_image = input_image.copy()

    if n_ids > 0:
        cv.aruco.drawDetectedMarkers(output_image, marker_corners, marker_ids)

        if estimate_pose:
            for i in range(n_ids):
                cv.drawFrameAxes(
                    output_image,
                    camera_matrix,
                    dist_coeffs,
                    rvecs[i],
                    tvecs[i],
                    n_markers * 1.5,
                    2,
                )

        if show_rejected and len(rejected_candidates):
            cv.aruco.drawDetectedMarkers(output_image, rejected_candidates, None)

        # Display the captured frame
        cv.imshow("Camera", output_image)
        cv.waitKey(0)


# def get_camera_parameters() -> tuple:
#     cv::Mat cameraMatrix, distCoeffs;
#     # You can read camera parameters from tutorial_camera_params.yml
#     readCameraParameters(filename, cameraMatrix, distCoeffs); // This function is located in detect_markers.cpp
#     std::vector<cv::Vec3d> rvecs, tvecs;
#     cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);
# return cameraMatrix, distCoeffs


def detect_aruco_camera(
    camera_id: int = 0,
    estimate_pose: bool = True,
    show_rejected: bool = True,
) -> None:
    # init aruco detector
    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    # init camera capture
    camera = cv.VideoCapture(camera_id)

    camera_matrix, dist_coeffs = read_parameters()
    rvecs: list[Any] = []
    tvecs: list[Any] = []

    # set coordinate system
    # obj_points = cv.Mat((4, 1))  # cv.CV_32FC3
    # objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    # Set coordinate system
    obj_points = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        dtype=np.float32,
    )

    while True:
        # get image from camera
        ret, image = camera.read()

        if not ret:
            logger.warning("failed to grab frame")
            break

        total_time: float = 0.0
        total_iterations: float = 0.0
        tick = cv.getTickCount()

        # detect markers and estimate pose
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(image)
        n_markers: int = len(marker_corners)
        n_ids: int = len(marker_ids)

        # calibration data from tutorial_camera_params.yml

        if estimate_pose and n_markers > 0:
            # Calculate pose for each marker
            for i in range(n_markers):
                ret, rvecs[i], tvecs[i] = cv.solvePnP(
                    obj_points,
                    camera_matrix,
                    dist_coeffs,
                    marker_corners[i],
                )
                if not ret:
                    logger.warning(f"Pose estimation failed for marker {marker_ids[i]}")

        current_time = (cv.getTickCount() - tick) / cv.getTickFrequency()
        total_time += current_time
        total_iterations += 1

        if total_iterations % 30 == 0:
            logger.info(
                f"Detection Time = {current_time * 1000} ms "
                f"(Mean = {1000 * total_time / total_iterations} ms)"
            )

        # draw results
        image_copy = image.copy()

        if n_ids > 0:
            cv.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)

            if estimate_pose:
                for i in range(n_ids):
                    cv.drawFrameAxes(
                        image_copy,
                        camera_matrix,
                        dist_coeffs,
                        rvecs[i],
                        tvecs[i],
                        n_markers * 1.5,
                        2,
                    )

        if show_rejected and len(rejected_candidates):
            cv.aruco.drawDetectedMarkers(image_copy, rejected_candidates, None)

        # Display the captured frame
        cv.imshow("Camera", image_copy)
        cv.waitKey(0)
        camera.release()
        cv.destroyAllWindows()
        del camera


if __name__ == "__main__":
    estimate_pose_aruco("images/aruco_tags_scene.jpg")

    # detect aruco tag in camera feed
    # detect_aruco_camera()
