import cv2
import numpy as np
from py5 import Sketch

# Define the size of the plane in meters
PLANE_WIDTH = 5.0
PLANE_HEIGHT = 3.0

# Define the size of the window in pixels
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600


# Function to detect ArUco tags and return their positions
def detect_aruco_positions(image):
    # Load the dictionary that was used to generate the markers.
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    # Initialize the detector parameters using default values
    parameters = cv2.aruco.DetectorParameters_create()
    # Detect the markers in the image
    corners, ids, _ = cv2.aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    positions = []
    if ids is not None:
        for corner in corners:
            # Calculate the center of the marker
            center = np.mean(corner[0], axis=0)
            positions.append(center)
    return positions


class ArucoSketch(Sketch):
    def __init__(self, aruco_positions):
        super().__init__()
        self.aruco_positions = aruco_positions

    def settings(self):
        self.size(WINDOW_WIDTH, WINDOW_HEIGHT)

    def setup(self):
        self.rect_mode(self.CENTER)

    def draw(self):
        self.background(255)
        self.stroke(0)
        self.fill(255, 0, 0)

        for pos in self.aruco_positions:
            # Convert the ArUco position to the plane coordinates
            x = (pos[0] / WINDOW_WIDTH) * PLANE_WIDTH
            y = (pos[1] / WINDOW_HEIGHT) * PLANE_HEIGHT
            # Draw the ArUco tag position
            self.ellipse(
                x * (WINDOW_WIDTH / PLANE_WIDTH),
                y * (WINDOW_HEIGHT / PLANE_HEIGHT),
                10,
                10,
            )


def processing(image_path):
    # Load the image
    image = cv2.imread(image_path)
    # Detect ArUco positions
    aruco_positions = detect_aruco_positions(image)
    # Create and run the sketch
    sketch = ArucoSketch(aruco_positions)
    sketch.run_sketch()


if __name__ == "__main__":
    # Replace 'path_to_image.jpg' with the path to your image file
    processing("path_to_image.jpg")
