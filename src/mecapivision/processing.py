from py5 import Sketch

# Define the size of the plane in meters
PLANE_WIDTH = 5.0
PLANE_HEIGHT = 3.0

# Define the size of the window in pixels
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

from .detection import aruco


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


def processing():
    image_path = "images/aruco_tags_scene.jpg"
    aruco_corners, _, _ = aruco.detect_aruco(image_path)
    aruco_positions = aruco.get_arucos_positions(aruco_corners)
    # Create and run the sketch
    sketch = ArucoSketch(aruco_positions)
    sketch.run_sketch()
