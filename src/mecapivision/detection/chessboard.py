from glob import glob

import cv2 as cv
import numpy as np
from cv2.typing import MatLike

from mecapivision.calibration.fake_chessboard import (
    analyse_pictures_for_calibration,
    calibrate_camera,
)


def detect_corners(filename: str) -> MatLike:
    print(f"detecting corners in {filename}")

    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    return img


# POSE estimation
def calib_cam():
    print("Calibrating camera...")
    objpoints, imgpoints = analyse_pictures_for_calibration()
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        "images/left14.jpg", objpoints, imgpoints
    )
    print(f"Ret: {ret}")
    np.savez("B.npz", mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)


def draw(img, corners, imgpts):
    print("drawing corners")
    corner = tuple(corners[0].ravel().astype("int32"))
    imgpts = imgpts.astype("int32")
    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


def pose_estimation():
    print("estimating pose")
    # Load previously saved data
    with np.load("B.npz") as X:
        mtx, dist, _, _ = [X[i] for i in ("mtx", "dist", "rvecs", "tvecs")]

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    axis = np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)

    for fname in glob("images/left*.jpg"):
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (7, 6), None)

        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Find the rotation and translation vectors.
            ret, rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)

            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

            img = draw(img, corners2, imgpts)
            cv.imshow("img", img)
            k = cv.waitKey(0) & 0xFF

            if k == ord("s"):
                cv.imwrite(fname[:6] + ".png", img)

    cv.destroyAllWindows()


if __name__ == "__main__":
    calib_cam()
    pose_estimation()
    cv.destroyAllWindows()
