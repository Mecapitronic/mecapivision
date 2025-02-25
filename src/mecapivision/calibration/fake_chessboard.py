# exercise to calibrate a camera using opencv, based on a tutorial
# uses a set of images of a chessboard to calibrate the camera
# and then undistort the images

# source https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html
# TODO inspi https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv

import glob

import cv2 as cv
import numpy as np

from ._utils import print_reprojection_error

DEFAULT_IMAGE = "images/left12.jpg"


def calibrate_fake_camera(image_path: str = DEFAULT_IMAGE) -> None:
    print("Calibrating camera...")
    objpoints, imgpoints = analyse_pictures_for_calibration()

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
    print(f"Ret: {ret}")

    for image_path in glob.glob("images/left*.jpg"):
        undistort_image(image_path, mtx, dist)

    print_reprojection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)


def analyse_pictures_for_calibration() -> tuple[list[np.ndarray], list[np.ndarray]]:
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob("images/left*.jpg")

    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(img, (7, 6), corners2, ret)
            cv.imshow("img", img)
            cv.waitKey(500)

    cv.destroyAllWindows()
    return objpoints, imgpoints


def undistort_image(image_path: str, mtx: np.ndarray, dist: np.ndarray) -> None:
    img = cv.imread(image_path)

    # Undistortion
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # method 1: undistort (the easiest way)
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # method 2: remapping (more difficult)
    # # undistort
    # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]  # noqa: E203

    cv.imshow("original", img)
    cv.imshow("calibrated", dst)
    cv.waitKey(0)
