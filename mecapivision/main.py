import cv2 as cv
from aruco import detect_aruco, detect_aruco_camera, get_aruco_tag, list_cameras
from chessboard import detect_corners
from cv2.typing import MatLike


def display_image(image_name: str, image: MatLike):
    cv.imshow(image_name, image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def plop():
    print("Hello from mecapivision!")

    chess = detect_corners("images/chessboard.jpg")
    display_image("chess", chess)

    display_image("aruco", get_aruco_tag(31))

    detect_aruco("images/aruco_tags_scene.jpg")


def main():
    list_cameras()
    detect_aruco_camera()


if __name__ == "__main__":
    main()
