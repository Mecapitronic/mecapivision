from copy import deepcopy
from typing import Sequence

import cv2 as cv
from cv2 import aruco
from cv2.typing import MatLike, Scalar


def print_aruco(aruco_id: int, size_in_pixels: int = 200):
    all_aruco_wards: aruco.Dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    markerImage: MatLike = aruco.generateImageMarker(
        all_aruco_wards, aruco_id, size_in_pixels
    )
    cv.imshow("marker23", markerImage)
    cv.waitKey(0)


aruco_tags = "images/aruco_tags_scene.jpg"


def detect_aruco(image_path: str = aruco_tags):
    inputImage: MatLike = cv.imread(image_path, cv.IMREAD_COLOR)

    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    result: tuple[
        Sequence[MatLike],
        MatLike,
        Sequence[MatLike],
    ] = detector.detectMarkers(inputImage)

    (markerCorners, markerIds, rejectedCandidates) = result

    print("results:\n")
    print(markerCorners)
    print(markerIds)
    print(rejectedCandidates)

    outputImage: MatLike = inputImage.copy()
    aruco.drawDetectedMarkers(outputImage, markerCorners, markerIds)

    cv.imshow("output", outputImage)
    cv.waitKey(0)


def detect_aruco_camera():
    # init aruco detector
    detector_params: cv.aruco.DetectorParameters = cv.aruco.DetectorParameters()
    dictionary: cv.aruco.Dictionary = cv.aruco.getPredefinedDictionary(
        cv.aruco.DICT_6X6_250
    )
    detector = cv.aruco.ArucoDetector(dictionary, detector_params)

    # init camera capture
    camId = 0
    camera = cv.VideoCapture(camId)

    # Get the default frame width and height
    frame_width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT))

    waitTime: int = 10

    # set coordinate system
    objPoints = cv.Mat((4, 1))  # cv.CV_32FC3
    # objPoints.ptr<Vec3f>(0)[0] = Vec3f(-markerLength/2.f, markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[1] = Vec3f(markerLength/2.f, markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[2] = Vec3f(markerLength/2.f, -markerLength/2.f, 0);
    # objPoints.ptr<Vec3f>(0)[3] = Vec3f(-markerLength/2.f, -markerLength/2.f, 0);

    while True:
        # get image from camera
        ret, image = camera.read()

        if not ret:
            print("failed to grab frame")
            break

        totalTime: float = 0
        totalIterations: float = 0
        tick = cv.getTickCount()

        # detect markers and estimate pose
        corners, ids, rejected = detector.detectMarkers(image)
        nMarkers: int = len(corners)
        nIds: int = len(ids)

        rvecs = deepcopy(nMarkers)
        tvecs = deepcopy(nMarkers)

        if estimatePose and nIds > 0:
            # Calculate pose for each marker
            for i in range(nMarkers):
                ret, one, two = cv.solvePnP(
                    objPoints, camMatrix, distCoeffs, corners[i], rvecs[i], tvecs[i]
                )
                if not ret:
                    print(f"Pose estimation failed for marker {ids[i]}")
                    continue

        currentTime = (cv.getTickCount() - tick) / cv.getTickFrequency()
        totalTime += currentTime
        totalIterations += 1

        if totalIterations % 30 == 0:
            print(
                f"Detection Time = {currentTime * 1000} ms (Mean = {1000 * totalTime / totalIterations} ms)"
            )

        # draw results
        imageCopy = image.copy()

        if nIds > 0:
            cv.aruco.drawDetectedMarkers(imageCopy, corners, ids)

            if estimatePose:
                for i in range(nIds):
                    cv.drawFrameAxes(
                        imageCopy,
                        camMatrix,
                        distCoeffs,
                        rvecs[i],
                        tvecs[i],
                        markerLength * 1.5,
                        2,
                    )

        if showRejected and len(rejected):
            cv.aruco.drawDetectedMarkers(
                imageCopy, rejected, noArray(), Scalar(100, 0, 255)
            )

        # Display the captured frame
        cv.imshow("Camera", imageCopy)
        cv.waitKey(0)
        camera.release()
        cv.destroyAllWindows()
        del camera
