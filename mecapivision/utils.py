from glob import glob

from cv2 import CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture


def list_cameras() -> list[str]:
    available_cameras: list[str] = []
    for cam in glob("/dev/video*"):
        camera = VideoCapture(cam)
        if not camera.isOpened():
            print(f"camera {cam} is not available")
        else:
            print(f"camera {cam} is available")
            frame_width = int(camera.get(CAP_PROP_FRAME_WIDTH))
            frame_height = int(camera.get(CAP_PROP_FRAME_HEIGHT))
            print(f"camera frame width: {frame_width}")
            print(f"camera frame height: {frame_height}")

            available_cameras.append(cam)
            camera.release()

    return available_cameras
