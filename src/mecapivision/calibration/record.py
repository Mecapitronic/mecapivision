from pathlib import Path

import click
import cv2 as cv
from loguru import logger

from .._utils import CANT_RECEIVE_FRAME, DEFAULT_NAME, PICTURES_FOLDER, get_last_camera, open_camera


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
    default=10,
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

    if nb_pictures_needed == 0:
        nb_pictures_needed = 1000000
    logger.info(f"Recording {nb_pictures_needed} pictures. Press 'r' to take a picture, 'q' to quit")

    logger.info(f"Opening camera {video}")
    camera = open_camera(video)

    Path(pictures_folder).mkdir(parents=True, exist_ok=True)
    nb_pictures_taken = 0

    while camera.isOpened() and nb_pictures_taken < nb_pictures_needed:
        ret, image = camera.read()

        if not ret:
            logger.error(CANT_RECEIVE_FRAME)
            break

        cv.imshow("captured picture", image)

        key = cv.waitKey(1) & 0xFF  # waitKey UNE SEULE FOIS
        if key == ord("r"):
            cv.imwrite(
                f"{pictures_folder}/{pictures_basename}_{nb_pictures_taken}.jpg", image
            )
            nb_pictures_taken += 1
            print(f"Picture taken: {nb_pictures_taken}   \r", end=" ")

        if key == ord("q"):
            break

        if nb_pictures_taken > nb_pictures_needed:
            break

    camera.release()
    cv.destroyAllWindows()

    logger.info(f"{nb_pictures_taken} pictures taken and saved in {pictures_folder}/")
