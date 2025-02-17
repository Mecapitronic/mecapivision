import os
import sys

from cv2 import imread

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from chessboard import detect_corners


class TestChessboard:
    def test_detect_corners(self):
        img = "images/chessboard.jpg"
        result = detect_corners(img)

        expected = imread("images/test_chessboard.jpg")
        assert result == expected
