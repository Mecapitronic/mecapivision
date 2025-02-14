import cv2 as cv
import numpy as np
from cv2 import aruco


def main():
    print("Hello from vision!")
    print_aruco()


def print_aruco(size_in_pixels=200):
    all_aruco_wards: aruco.Dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    markerImage = aruco.generateImageMarker(all_aruco_wards, 23, size_in_pixels)
    cv.imwrite("marker23.png", markerImage)


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
