# Fundamental matrix using 8 points
# [[ -9.01047758e-09   2.77077394e-06  -1.85608077e-03]
#  [  2.05331917e-06   7.27732510e-08  -1.36590441e-03]
#  [  2.80996733e-04  -1.75639180e-03   1.00000000e+00]]


import cameraLocalisation2D
import cv2
import numpy as np


def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = list(map(int, [0, -r[2] / r[1]]))
        x1, y1 = list(map(int, [c, -(r[2] + r[0] * c) / r[1]]))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

    return img1, img2


if __name__ == "__main__":
    # TEST BALL DETECTION
    try:
        fishEyeCameraTop = cameraLocalisation2D.FishEyeCamera(1)
    except:
        print("Failed to correctly set up the top camera")
        exit()

    try:
        fishEyeCameraSide= cameraLocalisation2D.FishEyeCamera(2)
    except:
        print("Failed to correctly set up the side camera")
        exit()

    numberOfOrangeBalls = 4
    numberOfBlueBalls = 4
    ballDetectorOrangeTop = cameraLocalisation2D.BallDetector(numberOfOrangeBalls)
    ballDetectorBlueTop = cameraLocalisation2D.BallDetector(numberOfBlueBalls)
    ballDetectorOrangeSide = cameraLocalisation2D.BallDetector(numberOfOrangeBalls)
    ballDetectorBlueSide = cameraLocalisation2D.BallDetector(numberOfBlueBalls)

    ballDetectorOrangeTop.setCalibration(31.7, 2.6*3)
    ballDetectorOrangeSide.setCalibration(31.7, 2.6*3)

    ballDetectorBlueTop.setCalibration(231.7, 6*3)
    ballDetectorBlueSide.setCalibration(231.7, 6*3)

    # ballDetectorBlueTop.calibrateColour(fishEyeCameraTop)
    # ballDetectorBlueTop.displayCalibration()
    #
    # ballDetectorBlueSide.calibrateColour(fishEyeCameraTop)
    # ballDetectorBlueSide.displayCalibration()
    # exit()

    print("Calibrate the initial positions of the orange balls.  This will need to be done twice, once per camera.")
    ballDetectorOrangeTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorOrangeSide.calibrateInitialLocations(fishEyeCameraSide)
    print("Calibrate the initial positions of the blue balls.  This will need to be done twice, once per camera.")
    ballDetectorBlueTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorBlueSide.calibrateInitialLocations(fishEyeCameraSide)
    cv2.destroyAllWindows()

    while True:
        undistortedFrameTop = fishEyeCameraTop.getUndistortedFrame()
        undistortedFrameSide = fishEyeCameraSide.getUndistortedFrame()

        maxDelta = 50
        ballDetectorOrangeTop.updateLocations(undistortedFrameTop, fishEyeCameraTop, maxDelta, debug=False)
        ballDetectorBlueTop.updateLocations(undistortedFrameTop, fishEyeCameraTop, maxDelta, debug=False)
        ballDetectorOrangeSide.updateLocations(undistortedFrameSide, fishEyeCameraSide, maxDelta, debug=False)
        ballDetectorBlueSide.updateLocations(undistortedFrameSide, fishEyeCameraSide, maxDelta, debug=False)
        topBalls = ballDetectorOrangeTop.getBallsPointsAsTuples()+ballDetectorBlueTop.getBallsPointsAsTuples()
        sideBalls = ballDetectorOrangeSide.getBallsPointsAsTuples()+ballDetectorBlueSide.getBallsPointsAsTuples()

        topBalls = np.int32(topBalls)
        sideBalls = np.int32(sideBalls)

        F, mask = cv2.findFundamentalMat(topBalls,sideBalls,cv2.FM_RANSAC)

        print("Fundamental Matrix:")
        print(F)
        print("Mask")
        print(mask)

        # Find epilines corresponding to points in right image (second image) and
        # drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(sideBalls.reshape(-1, 1, 2), 2, F)
        lines1 = lines1.reshape(-1, 3)
        img5, img6 = drawlines(undistortedFrameTop, undistortedFrameSide, lines1, topBalls, sideBalls)
        # drawing its lines on left image
        lines2 = cv2.computeCorrespondEpilines(topBalls.reshape(-1, 1, 2), 1, F)
        lines2 = lines2.reshape(-1, 3)
        img3, img4 = drawlines(undistortedFrameSide, undistortedFrameTop, lines2, sideBalls, topBalls)

        cv2.imshow("Processed Frame Top", undistortedFrameTop)
        cv2.imshow("Processed Frame Side", undistortedFrameSide)
        cv2.imshow("image 3", img3)
        cv2.imshow("image 4", img4)
        cv2.imshow("image 5", img5)
        cv2.imshow("image 6", img6)

        wk = cv2.waitKey(1)
        if wk != -1:
            print(wk)
        if wk == 1048689 or wk ==113: # q
            break

    cv2.destroyAllWindows()

