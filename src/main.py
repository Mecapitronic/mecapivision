import cv2 as cv

from aruco import detect_aruco, print_aruco
from chessboard import detect_corners


def main():
    print("Hello from vision!")

    chess = detect_corners()
    cv.imshow("chess", chess)
    cv.waitKey(0)

    print_aruco(31)
    detect_aruco()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
