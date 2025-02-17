import cv2 as cv

from mecapivision.chessboard import detect_corners


class TestChessboard:
    def test_detect_corners(self):
        expected = cv.imread("images/test_chessboard.tiff")

        img = "images/chessboard.jpg"
        result = detect_corners(img)

        assert result.shape == expected.shape

        difference = cv.subtract(result, expected)
        b, g, r = cv.split(difference)
        assert cv.countNonZero(b) == 0
        assert cv.countNonZero(g) == 0
        assert cv.countNonZero(r) == 0
