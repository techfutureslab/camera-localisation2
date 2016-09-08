import cameraLocalisation2D
import cv2
import numpy as np

def distace(A,B):
    # for i in range(len(A)):
    dx = A[0]-B[0]
    dy = A[1]-B[1]
    dz = A[2]-B[1]
    return (dx,dy,dz)

def roundTuple(A):
    rA = [0,0]
    for i in range(len(A)):
        rA[i] =int(A[i])
    return tuple(rA)

if __name__ == "__main__":
    # TEST BALL DETECTION
    fishEyeCameraTop = cameraLocalisation2D.FishEyeCamera(1)
    fishEyeCameraSide= cameraLocalisation2D.FishEyeCamera(2)

    ballDetectorOrangeTop = cameraLocalisation2D.BallDetector(1)
    ballDetectorBlueTop = cameraLocalisation2D.BallDetector(1)
    ballDetectorOrangeSide = cameraLocalisation2D.BallDetector(1)
    ballDetectorBlueSide = cameraLocalisation2D.BallDetector(1)

    ballDetectorOrangeTop.setCalibration(31.7, 2.6*3)
    ballDetectorBlueTop.setCalibration(31.7, 2.6*3) #231,6
    ballDetectorOrangeSide.setCalibration(31.7, 2.6*3)
    ballDetectorBlueSide.setCalibration(31.7, 2.6*3)

    ballDetectorOrangeTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorOrangeSide.calibrateInitialLocations(fishEyeCameraSide)
    ballDetectorBlueTop.calibrateInitialLocations(fishEyeCameraTop)
    ballDetectorBlueSide.calibrateInitialLocations(fishEyeCameraSide)
    cv2.destroyAllWindows()


    while True:
        undistortedFrameTop = fishEyeCameraTop.getUndistortedFrame()
        undistortedFrameSide = fishEyeCameraSide.getUndistortedFrame()

        maxDelta = 100
        ballDetectorOrangeTop.updateLocations(undistortedFrameTop, fishEyeCameraTop,maxDelta, debug=False)
        ballDetectorBlueTop.updateLocations(undistortedFrameTop, fishEyeCameraTop,maxDelta, debug=False)
        ballDetectorOrangeSide.updateLocations(undistortedFrameSide, fishEyeCameraSide,maxDelta, debug=False)
        ballDetectorBlueSide.updateLocations(undistortedFrameSide, fishEyeCameraSide,maxDelta, debug=False)
        topBalls = ballDetectorOrangeTop.getBallsPointsAsTuples()+ballDetectorBlueTop.getBallsPointsAsTuples()
        sideBalls = ballDetectorOrangeSide.getBallsPointsAsTuples()+ballDetectorBlueSide.getBallsPointsAsTuples()
        cv2.line(undistortedFrameTop,roundTuple(topBalls[0]),roundTuple(topBalls[1]),(255,0,0),3)
        cv2.line(undistortedFrameSide,roundTuple(sideBalls[0]),roundTuple(sideBalls[1]),(0,0,250),3)
        print ("Top"+str(topBalls))
        print ("Side"+str(sideBalls))

        cv2.imshow("Processed Frame Top", undistortedFrameTop)
        cv2.imshow("Processed Frame Side", undistortedFrameSide)

        wk = cv2.waitKey(1)
        if wk != -1:
            print(wk)
        if wk == 1048689 or wk == 113:  # q
            break

    cv2.destroyAllWindows()