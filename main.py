from typing import Sequence

import cv2 as cv
import numpy as np
from cv2 import aruco
from cv2.typing import MatLike

from .detect_aruco import detect_aruco_camera

# source https://docs.opencv.org/4.x/singlemarkerssource.jpg


def main():
    print("Hello from vision!")
    detect_aruco_camera()


def print_aruco(size_in_pixels=200):
    all_aruco_wards: aruco.Dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    markerImage: MatLike = aruco.generateImageMarker(
        all_aruco_wards, 23, size_in_pixels
    )
    cv.imwrite("marker23.png", markerImage)


def detect_aruco():
    inputImage: MatLike = cv.imread("multiple_aruco.jpg", cv.IMREAD_COLOR)

    detectorParams: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detectorParams)

    result: tuple[
        Sequence[MatLike],
        MatLike,
        Sequence[MatLike],
    ] = detector.detectMarkers(inputImage)

    (markerCorners, markerIds, rejectedCandidates) = result

    print("results:\n")
    print(markerCorners)
    print(markerIds)
    print(rejectedCandidates)

    outputImage: MatLike = inputImage.copy()
    aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)

    cv.imshow("output", outputImage)
    cv.waitKey(0)
    cv.destroyAllWindows()


def detect_corners():
    filename = "chessboard.jpg"
    img = cv.imread(filename)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    gray = np.float32(gray)
    dst = cv.cornerHarris(gray, 2, 3, 0.04)

    # result is dilated for marking the corners, not important
    dst = cv.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv.imshow("dst", img)
    if cv.waitKey(0) & 0xFF == 27:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
