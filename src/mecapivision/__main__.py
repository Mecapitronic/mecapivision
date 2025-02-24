from random import randint

import cv2 as cv
from cv2.typing import MatLike

from .detection.aruco import (
    detect_aruco,
    detect_aruco_camera,
    get_aruco_tag,
)
from .detection.chessboard import detect_corners


def display_image(image_name: str, image: MatLike):
    cv.imshow(image_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def main():
    print("Hello from mecapivision!")

    chess = detect_corners("images/chessboard.jpg")
    display_image("chessboard corners", chess)

    tag_id = randint(0, 100)
    display_image(f"aruco tag {tag_id}", get_aruco_tag(tag_id))
    detect_aruco("images/aruco_tags_scene.jpg")

    detect_aruco_camera()


if __name__ == "__main__":
    main()
