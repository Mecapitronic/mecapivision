"""
Calibrate the camera with a chessboard to undistort the image
Can calibrate with livestream or in two steps: record pictures then calibrate with an image set

Source:
    https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html

TODO:
    https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv
"""

import sys
from glob import glob
from typing import Sequence

import cv2 as cv
import numpy as np
from loguru import logger

from ._utils import (
    CANT_RECEIVE_FRAME,
    get_last_camera,
    print_reprojection_error,
    save_camera_calibration,
)
from .record import DEFAULT_NAME, PICTURES_FOLDER
from .undistort import undistort_image, undistort_livestream

NUM_CHESSBOARD_CORNERS_X = 6
NUM_CHESSBOARD_CORNERS_Y = 9


def calibrate_fake_camera() -> None:
    """exercise to calibrate a camera using opencv, based on a tutorial
    uses a set of images of a chessboard to calibrate the camera and then undistort the images

    Args:
        images_folder (str): folder containing the images of the chessboard
        image_path (str, optional): path to the image to undistort. Defaults to DEFAULT_IMAGE.
    """
    logger.info("Calibrating fake camera from pictures")
    images_folder: str = "images/"
    images_base_name: str = "left"
    images = glob(f"{images_folder}{images_base_name}*.jpg")

    objpoints, imgpoints, imgsize = analyse_chessboard_pictures(images)
    mtx, dist = calibrate_from_pictures(objpoints, imgpoints, imgsize)

    # display undistorted images
    for image_path in images:
        undistort_image(image_path, mtx, dist)


def calibrate_camera_from_pictures() -> None:
    """Calibrate the camera with a set of pictures of a chessboard.
    Pictures are recorded with the record_pictures function in record.py
    """
    logger.info("Calibrating camera from pictures")
    images = glob(f"{PICTURES_FOLDER}{DEFAULT_NAME}*.jpg")

    objpoints, imgpoints, imgsize = analyse_chessboard_pictures(images)
    mtx, dist = calibrate_from_pictures(objpoints, imgpoints, imgsize)

    save_camera_calibration("camera_calibration_pictures.npz", mtx, dist)


def calibrate_camera_from_livestream() -> None:
    logger.info("Calibrating camera with livestream")
    camera = get_last_camera()
    mtx, dist = calibrate_camera_live(camera)

    save_camera_calibration("camera_calibration_live.npz", mtx, dist)


# LIVE CALIBRATION


def calibrate_camera_live(camera: str) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Calibrating camera...")

    objpoints, imgpoints, imgsize = analyse_chessboards_live(camera)

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        imgsize,
        None,
        None,
    )  # type: ignore

    logger.info(f"Calibration result: {ret}")
    print_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    logger.info("starting undistorted livestream")
    undistort_livestream(camera, mtx, dist)

    return mtx, dist


def analyse_chessboards_live(
    video: str,
) -> tuple[list[np.ndarray], list[np.ndarray], Sequence[int]]:
    """Analyse multiple chessboard pictures to calibrate the camera

    Args:
        video (str): camera device file path in /dev

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], Sequence[int]]: object points, image points, image size
    """
    logger.info("Analyse chessboards live")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(
        (NUM_CHESSBOARD_CORNERS_X * NUM_CHESSBOARD_CORNERS_Y, 3), np.float32
    )
    objp[:, :2] = np.mgrid[
        0:NUM_CHESSBOARD_CORNERS_Y, 0:NUM_CHESSBOARD_CORNERS_X
    ].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    logger.info("opening camera to get chessboard pictures")
    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    logger.info(
        "Press 'r' to take a picture when the chessboard is detected. Press 'q' to quit."
    )
    nb_pictures_taken = 0
    while camera.isOpened():
        ret, image = camera.read()

        if not ret:
            logger.info(CANT_RECEIVE_FRAME)
            break

        cv.imshow("captured picture", image)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_FAST_CHECK
            + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        ret, corners = cv.findChessboardCorners(
            gray, (9, NUM_CHESSBOARD_CORNERS_X), flags=flags
        )

        # If found, add object points, image points (after refining them)
        if ret:
            logger.info("Chessboard found")

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv.drawChessboardCorners(image, (8, 7), corners2, ret)
            cv.imshow("detection", image)

            if cv.waitKey(20) & 0xFF == ord("r"):
                objpoints.append(objp)
                imgpoints.append(corners2)

                nb_pictures_taken += 1
                logger.info(f"Picture {nb_pictures_taken} taken")

        else:
            logger.warning("Chessboard not found")
            cv.imshow("detection", image)

        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    ret, image = camera.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    camera.release()
    cv.destroyAllWindows()

    logger.info(imgpoints)
    logger.info(objpoints)
    logger.info(f"got {len(imgpoints)} image points and {len(objpoints)} object points")

    return objpoints, imgpoints, image_size


# CALIBRATION WITH PICTURES


def calibrate_from_pictures(
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
    image_size: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the camera with a set of pictures of a chessboard

    Args:
        objpoints (list[np.ndarray]): object points
        imgpoints (list[np.ndarray]): image points
        image_size (Sequence[int]): size of the images

    Returns:
        tuple[np.ndarray, np.ndarray]: camera matrix and distortion coefficients
    """
    logger.info("Calibrating camera with data points")

    logger.info("nb points", len(objpoints), len(imgpoints))
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )  # type: ignore

    logger.info(f"Resultat: {ret}")
    print_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist


def analyse_chessboard_pictures(
    images_paths_list: list[str],
    display: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], Sequence[int]]:
    logger.info("Analysing chessboard pictures")

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(
        (NUM_CHESSBOARD_CORNERS_X * NUM_CHESSBOARD_CORNERS_Y, 3), np.float32
    )
    objp[:, :2] = np.mgrid[
        0:NUM_CHESSBOARD_CORNERS_Y, 0:NUM_CHESSBOARD_CORNERS_X
    ].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    image_size = []

    missing_images: int = 0
    for fname in images_paths_list:
        # spaces clear the line from previous print
        logger.debug(f"Analysing {fname}")

        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # usefull only once
        image_size = gray.shape[::-1]

        # Find the chess board corners
        flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_FAST_CHECK
            + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        ret, corners = cv.findChessboardCorners(
            gray,
            (
                NUM_CHESSBOARD_CORNERS_Y,
                NUM_CHESSBOARD_CORNERS_X,
            ),
            flags=flags,
        )

        if not ret:
            logger.warning(f"Chessboard not found in {fname}")
            missing_images += 1
            continue

        # If found, add object points, image points (after refining them)
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        if display:
            # Draw and display the corners
            cv.drawChessboardCorners(
                img,
                (
                    NUM_CHESSBOARD_CORNERS_Y,
                    NUM_CHESSBOARD_CORNERS_X,
                ),
                corners2,
                ret,
            )
            cv.imshow("img", img)
            cv.waitKey(500)

    if missing_images >= len(images_paths_list) * 0.2:
        logger.error(
            f"Cannot calibrate camera: {missing_images} images were not used because the chessboard was not found"
        )
        sys.exit(1)

    logger.info(f"Points found (image, IRL): {len(imgpoints)} {len(objpoints)}")
    cv.destroyAllWindows()
    return objpoints, imgpoints, image_size
