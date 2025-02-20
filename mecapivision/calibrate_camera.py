# source https://docs.opencv.org/4.11.0/dc/dbb/tutorial_py_calibration.html
# TODO inspi https://www.kaggle.com/code/danielwe14/stereocamera-calibration-with-opencv

import glob

import cv2 as cv
import numpy as np
from utils import list_cameras


def camera_calibration(video: int) -> None:
    print("Calibrating camera...")
    objpoints, imgpoints = analyse_multiple_chessboard_pictures(video)

    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        "images/left12.jpg", objpoints, imgpoints
    )
    print(f"Calibration result: {ret}")

    for img in glob.glob("images/left*.jpg"):
        undistort_image(img, mtx, dist)

    re_projection_error(objpoints, imgpoints, mtx, dist, rvecs, tvecs)


def analyse_multiple_chessboard_pictures(
    video: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    print("opening camera to get chessboard pictures")
    camera = cv.VideoCapture(video)
    while True:
        ret, image = camera.read()

        if not ret:
            print("Can't receive frame (stream end)")
            break

        cv.imshow("captured picture", image)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv.drawChessboardCorners(image, (7, 6), corners2, ret)
            cv.imshow("captured picture", image)

    cv.destroyAllWindows()
    return objpoints, imgpoints


def calibrate_camera(
    image_path: str,
    objpoints: list[np.ndarray],
    imgpoints: list[np.ndarray],
) -> tuple:
    img = cv.imread(image_path)

    # Calibration
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        None,
        None,
    )

    return ret, mtx, dist, rvecs, tvecs


def undistort_livestream(video: int) -> None:
    camera = cv.VideoCapture(0)

    while True:
        ret, image = camera.read()

        if not ret:
            print("Can't receive frame (stream end)")
            break

        # Undistortion
        h, w = img.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # method 1: undistort (the easiest way)
        dst = cv.undistort(img, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]  # noqa: E203

        cv.imshow("original", image)
        cv.imshow("calibrated", dst)


def undistort_image(img, mtx, dist):
    img = cv.imread(img)

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


def re_projection_error(
    objpoints: list[np.ndarray], imgpoints: list[np.ndarray], mtx, dist, rvecs, tvecs
):
    # Re-projection Error
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error
    print("total error: {}".format(mean_error / len(objpoints)))


def main():
    available_cameras = list_cameras()
    print(available_cameras)

    camera = cv.VideoCapture(available_cameras[-1])
    while True:
        ret, image = camera.read()
        cv.imshow("camera", image)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    camera.release()
    cv.destroyAllWindows()

    camera = int(0)
    camera_calibration(camera)
    undistort_livestream(camera)


if __name__ == "__main__":
    main()
