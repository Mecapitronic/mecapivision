import cv2 as cv
from numpy import ndarray

CANT_RECEIVE_FRAME = "Can't receive frame (stream end)"


def undistort_livestream(video: str, mtx, dist) -> None:
    camera = cv.VideoCapture(video)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, image = camera.read()

        if not ret:
            print(CANT_RECEIVE_FRAME)
            break

        # Undistortion
        h, w = image.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # method 1: undistort (the easiest way)
        dst = cv.undistort(image, mtx, dist, None, newcameramtx)
        # crop the image
        x, y, w, h = roi
        dst = dst[y : y + h, x : x + w]  # noqa: E203

        cv.imshow("original", image)
        cv.imshow("calibrated", dst)

        if cv.waitKey(20) & 0xFF == ord("q"):
            break

    camera.release()
    cv.destroyAllWindows()


def undistort_image(image_path: str, mtx: ndarray, dist: ndarray) -> None:
    img = cv.imread(image_path)

    # Undistortion
    h, w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # method 1: undistort (the easiest way)
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)

    # method 2: remapping (more difficult)
    # # undistort
    # mapx, mapy = cv.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)
    # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    x, y, w, h = roi
    dst = dst[y : y + h, x : x + w]  # noqa: E203

    cv.imshow("original", img)
    cv.imshow("calibrated", dst)
    cv.waitKey(0)
