"""
Calibrate the camera with a chessboard to undistort the image

Source:
    https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html

TODO:
    https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv
"""


# we might need to calibrate in two steps:
# 1. capture images of the chessboard
# 2. calibrate the camera with the images
#    2.1. record calibration parameters in a file
# 3. undistort livestream

from typing import Sequence

import cv2 as cv
import numpy as np

from ._utils import (
    get_last_camera,
    print_reprojection_error,
    save_camera_calibration,
)

CANT_RECEIVE_FRAME = "Can't receive frame (stream end)"


def calibrate_camera_with_chessboard():
    camera = get_last_camera()
    calibrate_camera(camera)


def calibrate_camera(camera: str) -> None:
    print("Calibrating camera...")

    objpoints, imgpoints, imgsize = analyse_multiple_chessboards_live(
        camera, save_pictures=True
    )

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


def analyse_multiple_chessboards_live(
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


def undistort_livestream(video: str, mtx, dist) -> None:
    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, image = camera.read()

        if not ret:
            print(CANT_RECEIVE_FRAME)
            break

        # Undistortion
        h, w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # method 1: undistort (the easiest way)
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]  # noqa: E203

        cv.imshow("original", image)
        cv.imshow("calibrated", dst)

        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()
