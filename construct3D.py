import cameraLocalisation2D
import cv2
import numpy as np


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


if __name__ == "__main__":
    # TEST BALL DETECTION
    fishEyeCameraTop = cameraLocalisation2D.FishEyeCamera(0)
    fishEyeCameraSide= cameraLocalisation2D.FishEyeCamera(1)
    ballDetectorOrangeTop = cameraLocalisation2D.BallDetector(4)
    ballDetectorBlueTop = cameraLocalisation2D.BallDetector(4)
    ballDetectorOrangeSide = cameraLocalisation2D.BallDetector(4)
    ballDetectorBlueSide = cameraLocalisation2D.BallDetector(4)

    ballDetectorOrangeTop.setCalibration(31.7, 2.6*3)
    ballDetectorBlueTop.setCalibration(231.7, 6*3)
    ballDetectorOrangeSide.setCalibration(31.7, 2.6*3)
    ballDetectorBlueSide.setCalibration(231.7, 6*3)
    # ballDetectorBlue.calibrateColour(fishEyeCameraTop)
    # ballDetectorBlue.displayCalibration()
    # exit()
    ballDetectorOrangeTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorOrangeSide.calibrateInitialLocations(fishEyeCameraSide)
    ballDetectorBlueTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorBlueSide.calibrateInitialLocations(fishEyeCameraSide)
    cv2.destroyAllWindows()

    while True:
        undistortedFrameTop = fishEyeCameraTop.getUndistortedFrame()
        undistortedFrameSide = fishEyeCameraSide.getUndistortedFrame()

        ballDetectorOrangeTop.updateLocations(undistortedFrameTop, fishEyeCameraTop, debug=False)
        ballDetectorBlueTop.updateLocations(undistortedFrameTop, fishEyeCameraTop, debug=False)
        ballDetectorOrangeSide.updateLocations(undistortedFrameSide, fishEyeCameraSide, debug=False)
        ballDetectorBlueSide.updateLocations(undistortedFrameSide, fishEyeCameraSide, debug=False)
        topBalls = ballDetectorOrangeTop.getBallsPointsAsTuples()+ballDetectorBlueTop.getBallsPointsAsTuples()
        sideBalls = ballDetectorOrangeSide.getBallsPointsAsTuples()+ballDetectorBlueSide.getBallsPointsAsTuples()
        F, mask = cv2.findFundamentalMat(topBalls,sideBalls,cv2.FM_8POINT)

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(sideBalls.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(undistortedFrameTop, undistortedFrameSide, lines1, topBalls, sideBalls)

        cv2.imshow("Processed Frame Top", undistortedFrameTop)
        cv2.imshow("Processed Frame Side", undistortedFrameSide)
        cv2.imshow("image 5", img5)
        cv2.imshow("image 6", img6)

        wk = cv2.waitKey(1)
        if wk != -1:
            print wk
        if wk == 1048689: # q
            break

    cv2.destroyAllWindows()
    exit()
