import cv2 as cv

from aruco import detect_aruco, print_aruco
from camera_calibration import camera_calibration
from chessboard import detect_corners


def main():
    print("Hello from vision!")
    detect_corners()
    print_aruco(31)
    detect_aruco()
    cv.destroyAllWindows()


if __name__ == "__main__":
    camera_calibration()
