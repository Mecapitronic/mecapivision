"""
Calibrate the camera with a chessboard to undistort the image
Can calibrate with livestream or in two steps: record pictures then calibrate with an image set

Source:
    https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html

TODO:
    https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv
"""

from glob import glob
from typing import Sequence

import cv2 as cv
import numpy as np

from ._utils import (
    get_last_camera,
    print_reprojection_error,
    save_camera_calibration,
)
from .undistort import undistort_image, undistort_livestream

CANT_RECEIVE_FRAME = "Can't receive frame (stream end)"


def calibrate_camera_with_chessboard(live: bool = False) -> None:
    """Calibrate the camera with a chessboard. Can operate on livestream or with recorded images
    set live to True to calibrate with a livestream. Otherwise, the camera will take pictures of the chessboard

    Args:
        live (bool, optional): whether to calibrate with a livestream or recorded images.
        True is livestream, False is recorded images. Defaults to False.
    """
    if live:
        camera = get_last_camera()
        calibrate_camera(camera)

    else:
        print("Calibrating camera with recorded images")


def calibrate_fake_camera() -> None:
    """exercise to calibrate a camera using opencv, based on a tutorial
    uses a set of images of a chessboard to calibrate the camera and then undistort the images

    Args:
        images_folder (str): folder containing the images of the chessboard
        image_path (str, optional): path to the image to undistort. Defaults to DEFAULT_IMAGE.
    """
    print("Calibrating fake camera from pictures")
    images_folder: str = "images/"
    images_base_name: str = "left"
    images = glob(f"{images_folder}{images_base_name}*.jpg")

    objpoints, imgpoints = analyse_chessboard_pictures(images)
    mtx, dist = calibrate_from_pictures(images[0], objpoints, imgpoints)

    # display undistorted images
    for image_path in images:
        undistort_image(image_path, mtx, dist)


# LIVE CALIBRATION


def calibrate_camera(camera: str) -> None:
    print("Calibrating camera...")

    objpoints, imgpoints, imgsize = analyse_chessboards_live(camera, save_pictures=True)

    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        imgsize,
        None,
        None,
    )  # type: ignore

    print(f"Calibration result: {ret}")
    save_camera_calibration("camera_calibration.npz", mtx, dist)
    print_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    print("starting undistorted livestream")
    undistort_livestream(camera, mtx, dist)


def analyse_chessboards_live(
    video: str,
    *,
    save_pictures: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], Sequence[int]]:
    """Analyse multiple chessboard pictures to calibrate the camera

    Args:
        video (str): camera device file path in /dev

    Returns:
        tuple[list[np.ndarray], list[np.ndarray], Sequence[int]]: object points, image points, image size
    """
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    print("opening camera to get chessboard pictures")
    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    print(
        "Press 'r' to take a picture when the chessboard is detected. Press 'q' to quit."
    )
    nb_pictures_taken = 0
    while camera.isOpened():
        ret, image = camera.read()

        if not ret:
            print(CANT_RECEIVE_FRAME)
            break

        cv.imshow("captured picture", image)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            + cv.CALIB_CB_FAST_CHECK
            + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        ret, corners = cv.findChessboardCorners(gray, (9, 6), flags=flags)

        # If found, add object points, image points (after refining them)
        if ret:
            print("Chessboard found")

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Draw and display the corners
            cv.drawChessboardCorners(image, (8, 7), corners2, ret)
            cv.imshow("detection", image)

            if cv.waitKey(20) & 0xFF == ord("r"):
                objpoints.append(objp)
                imgpoints.append(corners2)

                nb_pictures_taken += 1
                print(f"Picture {nb_pictures_taken} taken")
                if save_pictures:
                    cv.imwrite(f"my_calib/chessboard_{nb_pictures_taken}.jpg", image)
        else:
            print("Chessboard not found")
            cv.imshow("detection", image)

        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    ret, image = camera.read()
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    image_size = gray.shape[::-1]

    camera.release()
    cv.destroyAllWindows()

    print(imgpoints)
    print(objpoints)
    print(f"got {len(imgpoints)} image points and {len(objpoints)} object points")

    return objpoints, imgpoints, image_size


# CALIBRATION WITH PICTURES


def calibrate_from_pictures(
    image_path: str, objpoints, imgpoints
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate the camera with a set of pictures of a chessboard

    Args:
        image_path (str): path to the image to calibrate the camera

    Returns:
        tuple[np.ndarray, np.ndarray]: camera matrix and distortion coefficients
    """
    print("Calibrating camera from pictures")

    # Calibration
    img = cv.imread(image_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
    )  # type: ignore

    print(f"Resultat: {ret}")
    print_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)

    return mtx, dist


def analyse_chessboard_pictures(
    images_paths_list: list[str],
    display: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    for fname in images_paths_list:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            if display:
                # Draw and display the corners
                cv.drawChessboardCorners(img, (7, 6), corners2, ret)
                cv.imshow("img", img)
                cv.waitKey(500)

    cv.destroyAllWindows()
    return objpoints, imgpoints
