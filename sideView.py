import cv2
import cameraLocalisation2D
import gridMaker



if __name__ == "__main__":
    fishEyeCameraSide = cameraLocalisation2D.FishEyeCamera(deviceID=0)

    ballDetectorBlueSide = cameraLocalisation2D.BallDetector(1)
    ballDetectorBlueSide.setCalibration(231.7, 6*3)
    ballDetectorBlueSide.calibrateInitialLocations(fishEyeCameraSide)

    ballDetectorOrangeSide = cameraLocalisation2D.BallDetector(1)
    ballDetectorOrangeSide.setCalibration(31.7, 2.6*3)
    ballDetectorOrangeSide.calibrateInitialLocations(fishEyeCameraSide)


    while True:
        undistortedFrameSide = fishEyeCameraSide.getUndistortedFrame()
        ballDetectorOrangeSide.updateLocations(undistortedFrameSide, fishEyeCameraSide,50, debug=False)
        sideBalls = ballDetectorBlueSide.getBallsPointsAsTuples()+ballDetectorBlueSide.getBallsPointsAsTuples()
        print ("Side"+str(sideBalls))

        ballDetectorBlueSide.updateLocations(undistortedFrameSide, fishEyeCameraSide,50, debug=False)
        sideBalls = ballDetectorBlueSide.getBallsPointsAsTuples()+ballDetectorBlueSide.getBallsPointsAsTuples()
        print ("Side"+str(sideBalls))
        cv2.imshow("Processed Frame Side", undistortedFrameSide)

        wk = cv2.waitKey(1)
        if wk != -1:
            print(wk)
        if wk == 1048689 or wk == 113:  # q
            break

    cv2.destroyAllWindows()