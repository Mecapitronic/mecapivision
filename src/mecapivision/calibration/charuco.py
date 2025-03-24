"""Camera calibration using Charuco board.
Source: https://medium.com/@nflorent7/a-comprehensive-guide-to-camera-calibration-using-charuco-boards-and-opencv-for-perspective-9a0fa71ada5f
"""

import json
import os

import click
import cv2
import numpy as np
from loguru import logger

from .._settings import calibration_folder

# CONSTANTS
ARUCO_DICT = cv2.aruco.DICT_6X6_250  # Dictionary ID
SQUARES_VERTICALLY = 7  # Number of squares vertically
SQUARES_HORIZONTALLY = 5  # Number of squares horizontally
SQUARE_LENGTH = 40  # Square side length (in pixels)
MARKER_LENGTH = 20  # ArUco marker side length (in pixels)
MARGIN_PX = 20  # Margins size (in pixels)


def create_and_save_new_board(output_name: str = "charuco_marker.png") -> None:
    """Create and save a new Charuco board in order to calibrate the camera.
    You can also use a ChArUco generator such as calib.io
    """
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    size_ratio = SQUARES_HORIZONTALLY / SQUARES_VERTICALLY
    logger.info(f"Board size ratio: {size_ratio}")
    IMG_SIZE = tuple(
        i * SQUARE_LENGTH + 2 * MARGIN_PX
        for i in (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY)
    )
    img = cv2.aruco.CharucoBoard.generateImage(board, IMG_SIZE, marginSize=MARGIN_PX)
    cv2.imwrite(output_name, img)


def get_calibration_parameters(
    img_dir: str, show_img: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Get the calibration parameters from a set of images of a Charuco board

    Args:
        img_dir (str): folder where to find pictures taken with the camera to calibrate

    Returns:
        tuple[np.ndarray, np.ndarray]: camera matrix and distortion coefficients
    """
    logger.info("Getting calibration parameters from Charuco board")
    # Define the aruco dictionary, charuco board and detector
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    charucodetector = cv2.aruco.CharucoDetector(board)

    # Load images from directory
    image_files = [
        os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")
    ]
    all_charuco_ids = []
    all_charuco_corners = []

    # Loop over images and extraction of corners
    for image_file in image_files:
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_size = image.shape

        # old way
        # marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
        # ret, charucoCorners, charucoIds = cv2.interpolateCornersCharuco(marker_corners, marker_ids, image, board)
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            charucodetector.detectBoard(image)
        )

        # If at least one marker is detected
        if marker_ids is not None:
            logger.debug("Markers detected")

            if show_img:
                image_copy = image.copy()
                cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
                cv2.imshow("Markers", image_copy)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

            logger.debug(
                f"Charuco ID: {charuco_ids}; Charuco corners: {charuco_corners}"
            )
            if charuco_ids is not None and len(charuco_corners) > 3:
                logger.debug("Charuco corners found")
                all_charuco_corners.append(charuco_corners)
                all_charuco_ids.append(charuco_ids)

    logger.debug(f"Charuco corners: {all_charuco_corners}")
    logger.debug(f"Charuco ids: {all_charuco_ids}")
    # Calibrate camera with extracted information
    # Now that we've seen all of our images, perform the camera calibration
    # based on the set of points we've discovered
    result, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        charucoCorners=all_charuco_corners,
        charucoIds=all_charuco_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        rvecs=None,
        tvecs=None,
    )

    logger.info(f"Calibration result: {result}")
    logger.debug(
        f"Camera matrix:\n{camera_matrix}\nDistortion coefficients:\n{dist_coeffs}"
    )
    return camera_matrix, dist_coeffs


def save_calibration_to_json(json_file_path: str = "calibration.json"):
    logger.info("Saving calibration data to JSON file")
    SENSOR = "monochrome"
    LENS = "kowa_f12mm_F1.8"

    mtx, dist = get_calibration_parameters(img_dir="./images/")
    data = {"sensor": SENSOR, "lens": LENS, "mtx": mtx.tolist(), "dist": dist.tolist()}

    with open(json_file_path, "w") as json_file:
        json.dump(data, json_file, indent=4)

    logger.info(f"Data has been saved to {json_file_path}")


def load_calibration(
    json_file_path: str = "./calibration.json",
) -> tuple[np.ndarray, np.ndarray]:
    logger.info("Loading calibration data from JSON file")
    with open(json_file_path, "r") as file:  # Read the JSON file
        json_data = json.load(file)

    mtx = np.array(json_data["mtx"])
    dst = np.array(json_data["dist"])

    return mtx, dst


def undistort_image(image_path: str, mtx, dst) -> np.ndarray:
    """Undistort an image using the calibration parameters

    Returns:
        np.ndarray: undistorted image
    """
    logger.info(f"Undistorting image {image_path}")
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    h, w = image.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCamera_matrix(mtx, dst, (w, h), 1, (w, h))
    image = cv2.undistort(image, mtx, dst, None, newcameramtx)

    return image


def get_charucos_positions(image: np.ndarray, mtx: np.ndarray, dst: np.ndarray) -> None:
    logger.info("Getting Charuco board position")
    all_charuco_ids = []
    all_charuco_corners = []

    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_VERTICALLY, SQUARES_HORIZONTALLY),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        dictionary,
    )
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, params)

    marker_corners, marker_ids, rejectedCandidates = detector.detectMarkers(image)
    if (
        marker_ids is not None and len(marker_ids) > 0
    ):  # If at least one marker is detected
        # cv2.aruco.drawDetectedMarkers(image_copy, marker_corners, marker_ids)
        ret, charucoCorners, charucoIds = cv2.aruco.interpolateCornersCharuco(
            marker_corners, marker_ids, image, board
        )
        if (
            charucoCorners is not None
            and charucoIds is not None
            and len(charucoCorners) > 3
        ):
            all_charuco_corners.append(charucoCorners)
            all_charuco_ids.append(charucoIds)

        # rvec is a 1D numpy array representing the rotation vector that defines the 3D rotation of the ChArUco board.
        # tvec is a 1D numpy array that describes the translation of the ChArUco board in the camera’s coordinate system.
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            np.array(all_charuco_corners)[0],
            np.array(all_charuco_ids)[0],
            board,
            np.array(mtx),
            np.array(dst),
            np.empty(1),
            np.empty(1),
        )

        Zx, Zy, Zz = tvec[0][0], tvec[1][0], tvec[2][0]
        fx, fy = mtx[0][0], mtx[1][1]

        logger.debug(f"Zx: {Zx}; Zy: {Zy}; Zz: {Zz}")
        logger.debug(f"fx: {fx}; fy: {fy}")
        logger.info(f"Zz = {Zz}\nfx = {fx}")


def perspective_function(x, Z, f):
    return x * Z / f


@click.command()
@click.option(
    "--img_dir", default=calibration_folder, help="Directory containing images"
)
def calibrate_charuco(img_dir: str) -> None:
    logger.info("Calibrating camera using Charuco board")

    _, _ = get_calibration_parameters(img_dir, show_img=True)
    save_calibration_to_json(json_file_path="calibration.json")
    mtx, dst = load_calibration(json_file_path="calibration.json")
    image = undistort_image("my_calib/charuco14.jpg", mtx, dst)
    get_charucos_positions(image, mtx, dst)
