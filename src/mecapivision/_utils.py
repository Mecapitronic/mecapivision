from glob import glob
from pathlib import Path

import cv2 as cv
from loguru import logger
from numpy import load, ndarray, savez

from ._settings import camera_calibration_file

CANT_RECEIVE_FRAME = "Can't receive frame (stream end)"

PICTURES_FOLDER = "my_calib"
DEFAULT_NAME = "chessboard"


def list_cameras() -> list[str]:
    available_cameras: list[str] = []
    for cam in glob("/dev/video*"):
        camera = cv.VideoCapture(cam)
        if not camera.isOpened():
            logger.debug(f"camera {cam} is not available")
        else:
            logger.debug(f"camera {cam} is available")
            frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
            frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))
            logger.debug(f"camera frame width: {frame_width}")
            logger.debug(f"camera frame height: {frame_height}")

            available_cameras.append(cam)
            camera.release()

    return available_cameras


def get_last_camera() -> str:
    """Get the last camera available, the first one in the list is the built-in camera
    in laptops, if we have an external camera, it will be the last one.

    Returns:
        str: camera device file path in /dev
    """
    available_cameras = list_cameras()
    available_cameras.sort()
    logger.debug(available_cameras)
    return available_cameras[-1]


def print_calibration_result(mtx, dist) -> None:
    # Print the calibration results
    logger.info("Camera Matrix:\n", mtx)
    logger.info("Distortion Coefficients:\n", dist)


def save_camera_calibration(
    mtx: ndarray,
    dist: ndarray,
    file: str = camera_calibration_file,
) -> None:
    """Save camera calibration parameters to file using numpy

    Intrinsic parameters are the internal characteristics of your camera — think of them as the camera's personality traits.
    They include the focal length (which affects how zoomed in or out your images appear), the optical center (the point in the image where light rays converge), and distortion coefficients (which account for those pesky distortions like barrel or pincushion effects).
    Essentially, intrinsic parameters define how your camera sees the world.

    extrinsic parameters are all about the camera's position and orientation in space — basically, where your camera is and where it's looking.
    These are particularly important if you're using your camera in a multi-camera setup or for something like augmented reality, where aligning the real and virtual worlds is key.

    see https://medium.com/@amit25173/opencv-camera-calibration-03d19f0f52bc

    Args:
        file (str): file path where to store the calibration
        mtx (np.ndarray): camera matrix
        dist (np.ndarray): distortion coefficients
        rvecs (np.ndarray): rotation vectors
        tvecs (np.ndarray): translation
    """
    savez(file, mtx=mtx, dist=dist)
    logger.info(f"Calibration saved to {file}")


def load_camera_calibration(file: str) -> tuple[ndarray, ndarray]:
    """Load camera calibration parameters from file using numpy

    Args:
        file (str): file path where to load the calibration

    Returns:
        tuple[ndarray, ndarray]: camera matrix and distortion coefficients
    """
    # Later, load the calibration results
    with load(file) as data:
        mtx = data["mtx"]
        dist = data["dist"]

    return mtx, dist


def read_parameters(filename: str = "images/tutorial_camera_params.yml") -> tuple:
    # Read camera parameters from tutorial_camera_params.yml
    assert Path(filename).exists(), f"file {filename} does not exist"
    fs = cv.FileStorage(filename, cv.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("cameraMatrix").mat()
    dist_coeffs = fs.getNode("distCoeffs").mat()
    fs.release()
    return camera_matrix, dist_coeffs


def print_reprojection_error(
    objpoints: list[ndarray],
    imgpoints: list[ndarray],
    mtx,
    dist,
    rvecs,
    tvecs,
):
    """Print the re-projection error
    Re-projection error gives a good estimation of just how exact the found parameters are.
    The closer the re-projection error is to zero, the more accurate the parameters we found are.

    Args:
        objpoints (list[ndarray]): _description_
        imgpoints (list[ndarray]): _description_
        mtx (_type_): _description_
        dist (_type_): _description_
        rvecs (_type_): _description_
        tvecs (_type_): _description_
    """
    mean_error = 0.0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
        mean_error += error

    logger.info(
        "total error (closer to 0 is better): {}".format(mean_error / len(objpoints))
    )


def record_camera_stream():
    # Open the default camera
    cam = cv.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter("output.mp4", fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        if not ret:
            logger.error(CANT_RECEIVE_FRAME)
            break
        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv.imshow("Camera", frame)

        # Press 'q' to exit the loop
        if cv.waitKey(1) == ord("q"):
            break

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv.destroyAllWindows()
