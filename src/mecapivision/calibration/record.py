from pathlib import Path

import click
import cv2 as cv
from loguru import logger

from .._utils import CANT_RECEIVE_FRAME, DEFAULT_NAME, PICTURES_FOLDER, get_last_camera


@click.command()
@click.option(
    "--pictures_folder",
    "-f",
    default=PICTURES_FOLDER,
    help="Folder to save the pictures",
)
@click.option(
    "--pictures_basename",
    "-n",
    default=DEFAULT_NAME,
    help="Base name for the pictures",
)
@click.option(
    "--nb_pictures_needed",
    "-p",
    default=0,
    help="Number of pictures needed",
)
def record_pictures_cli(
    nb_pictures_needed: int, pictures_folder: str, pictures_basename: str
) -> None:
    record_pictures(
        get_last_camera(), pictures_folder, pictures_basename, nb_pictures_needed
    )


def record_pictures(
    video: str,
    pictures_folder: str,
    pictures_basename: str,
    nb_pictures_needed: int,
) -> None:
    logger.info("Recording pictures. Press 'r' to take a picture, 'q' to quit")

    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    Path(pictures_folder).mkdir(parents=True, exist_ok=True)
    nb_pictures_taken = 0
    if nb_pictures_needed == 0:
        nb_pictures_needed = 1000000

    while camera.isOpened():
        ret, image = camera.read()

        if not ret:
            logger.error(CANT_RECEIVE_FRAME)
            break

        cv.imshow("captured picture", image)

        if cv.waitKey(10) & 0xFF == ord("r"):
            cv.imwrite(
                f"{pictures_folder}/{pictures_basename}_{nb_pictures_taken}.jpg", image
            )
            nb_pictures_taken += 1
            print(f"Picture taken: {nb_pictures_taken}   \r", end=" ")

        if cv.waitKey(10) & 0xFF == ord("q"):
            break

        if nb_pictures_taken == nb_pictures_needed:
            break

    camera.release()
    cv.destroyAllWindows()

    logger.info(f"{nb_pictures_taken} pictures taken")
    logger.info(f"pictures saved in {pictures_folder}/{pictures_basename}0.jpg")
