import cv2 as cv

from mecapivision.detection.chessboard import detect_corners

images_folder = "images/"


class TestChessboard:
    def test_detect_corners(self):
        expected = cv.imread(f"{images_folder}test_chessboard.tiff")

        img = f"{images_folder}chessboard.jpg"
        result = detect_corners(img)

        assert result.shape == expected.shape

        difference = cv.subtract(result, expected)
        b, g, r = cv.split(difference)
        assert cv.countNonZero(b) == 0
        assert cv.countNonZero(g) == 0
        assert cv.countNonZero(r) == 0
