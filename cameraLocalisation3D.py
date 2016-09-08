import cameraLocalisation2D
import cv2 as cv


if __name__ == "__main__":
    # TEST BALL DETECTION
    fishEyeCamera = cameraLocalisation2D.FishEyeCamera(1)
    ballDetector = cameraLocalisation2D.BallDetector(4)

    ballDetector.setCalibration(31.7, 2.6*3)
    ballDetector.calibrateInitialLocations(fishEyeCamera)

    while True:
        undistortedFrame = fishEyeCamera.getUndistortedFrame()
        ballDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)
        cv.imshow("Processed Frame", undistortedFrame)

        wk = cv.waitKey(1)
        if wk != -1:
            print(wk)
        if wk == 1048689: # q
            break

    cv.destroyAllWindows()
    exit()
