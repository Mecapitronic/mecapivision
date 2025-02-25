from pathlib import Path

import cv2 as cv

from ._utils import get_last_camera

CANT_RECEIVE_FRAME = "Can't receive frame (stream end)"
PICTURES_FOLDER = "my_calib/"
DEFAULT_NAME = "chessboard"


def record_pictures_cli() -> None:
    record_pictures(get_last_camera())


def record_pictures(
    video: str,
    pictures_folder: str = PICTURES_FOLDER,
    pictures_basename: str = DEFAULT_NAME,
    nb_pictures_needed: int = 10,
) -> None:
    print("Recording pictures. Press 'r' to take a picture, 'q' to quit")

    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    Path(pictures_folder).mkdir(parents=True, exist_ok=True)
    nb_pictures_taken = 0

    while camera.isOpened():
        ret, image = camera.read()

        if not ret:
            print(CANT_RECEIVE_FRAME)
            break

        cv.imshow("captured picture", image)

        if cv.waitKey(10) & 0xFF == ord("r"):
            cv.imwrite(
                f"{pictures_folder}{pictures_basename}_{nb_pictures_taken}.jpg", image
            )
            nb_pictures_taken += 1
            print(f"Picture {nb_pictures_taken} taken")

        if cv.waitKey(10) & 0xFF == ord("q"):
            break

        if nb_pictures_taken == nb_pictures_needed:
            break

    camera.release()
    cv.destroyAllWindows()

    print(f"{nb_pictures_taken} pictures taken")
    print(f"pictures saved in {pictures_folder} as {pictures_basename}_*.jpg")
