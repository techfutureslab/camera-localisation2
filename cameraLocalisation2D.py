import cv2 as cv
import numpy as np
import time
import math
import colorsys


def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        # cv.circle(img,(x,y),100,(255,0,0),-1)
        # print hsvFrame[y,x]
        pass


class FishEyeCamera:
    def __init__(self, deviceID=0):
        # The fish-eye cameras support 720p and their angle-of-view is affected by the resolution
        self.height = 720
        self.width = 1280

        # Open camera
        self.captureVideo(deviceID)

        # Camera calibration parameters obtained 2016-08-11 (by Mohsen and Matthew) using captureImages.py and calibrate.py
        self.cameraMatrix = np.array([[416.25456303, 0., 663.64459394], [0., 387.2264034, 380.49903696], [0., 0., 1.]])
        self.distortionCoefficients = np.array([2.00198060e-01, -2.28216265e-01, 2.26068631e-04, -5.00177586e-04, 5.84619782e-02])
        self.newCameraMatrix, _ = cv.getOptimalNewCameraMatrix(self.cameraMatrix,
                                                               self.distortionCoefficients,
                                                               (self.width, self.height),
                                                               1,
                                                               (self.width, self.height))

        # TODO: The calibration parameters should really allow the undistorted image to be in real-world units.
        # For the meantime, we need to be able to convert from pixels to real-world units
        self.pixelsPerMillimeter = 0.5 # As at 2016-08-22

    def captureVideo(self, deviceID=0):
        '''Opens the camera (specified by deviceID).  Sets the resolution to 720p (1280 x 720).'''
        self.camera = cv.VideoCapture(deviceID)
        self.camera.set(cv.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv.CAP_PROP_FRAME_HEIGHT, self.height)

        ret, frame = self.camera.read()
        h, w = frame.shape[:2]
        assert ret
        assert h == self.height and w == self.width

    def getFrame(self):
        '''Returns a frame of video flipped in both horizontal and vertical axes.'''
        ret, frame = self.camera.read()
        frame = cv.flip(cv.flip(frame, 1), -1)
        return frame

    def getUndistortedFrame(self):
        frame = self.getFrame()
        undistortedFrame = cv.undistort(frame, self.cameraMatrix, self.distortionCoefficients, None, self.newCameraMatrix)
        return undistortedFrame


class RobotDetector:
    def __init__(self, numberOfColours = 3):
        self.numberOfColours = numberOfColours

        # Setup SimpleBlobDetector parameters.
        def createDetectorParameters():
            params = cv.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 10
            params.maxThreshold = 200

            # Filter by Colour
            params.filterByColor = True
            params.blobColor = 255

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 150

            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1

            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.87

            # Filter by Inertia
            params.filterByInertia = False
            params.minInertiaRatio = 0.01

            return params

        # Create SimpleBlobDetector (assumes OpenCV version 3)
        params = createDetectorParameters()
        self.detector = cv.SimpleBlobDetector_create(params)
        self.colourHuesDegrees = [hue for hue in range(0,255,255/self.numberOfColours)]
        self.colourHeusBandDegrees = [10]*numberOfColours

        class Robot:
            def __init__(self, colours):
                self.colours = colours
                self.location = None
                self.orientation = None

            def __str__(self):
                return "colors: "+str(self.colours)+" location: "+str(self.location)+" orientation: "+str(self.orientation)

        # Calculate robot indicies
        assert (numberOfColours >= 2)
        self.robots = []
        for c1 in xrange(numberOfColours):
            for c2 in xrange(c1+1, numberOfColours):
                self.robots.append(Robot((c1,c2)))



    def getMeanAndStdDevFromColourSamples(self, colourSamples):
        '''From a list of colourSamples, find the mean and standard deviation using a delta from the first sampled
        colour'''
        assert len(colourSamples) > 0

        # Source colour is the first of the colour samples
        sourceColour = int(colourSamples[0])

        # Go through the samples and delta between the current sample and the source.  There are two deltas, one in each
        # direction (possibly looping over 360 degrees).
        deltaUpColourSamples = [(int(x) - sourceColour)%256 for x in colourSamples]
        deltaDownColourSamples = [-((sourceColour - (int(x)))%256) for x in colourSamples]

        # Find the minimum of the two deltas
        deltaColourSamples = []
        for i in xrange(len(deltaDownColourSamples)):
            if abs(deltaDownColourSamples[i]) <= abs(deltaUpColourSamples[i]):
                deltaColourSamples.append(deltaDownColourSamples[i])
            else:
                deltaColourSamples.append(deltaUpColourSamples[i])

        # Find the mean and standard deviation of the (minimum) deltas
        meanDelta = float(sum(deltaColourSamples)) / len(deltaColourSamples)

        # Calculate the mean colour.
        meanColour = (sourceColour + meanDelta) % 256

        sdDelta = math.sqrt(float(sum([(x - meanDelta) ** 2 for x in deltaColourSamples])) / len(deltaColourSamples))

        # Convert the colours (from uint8) to degrees
        meanColourDegrees = float(meanColour) / 255 * 360
        sdColourDegrees = float(sdDelta) / 255 * 360

        return (meanColourDegrees, sdColourDegrees)

    def calibrateColour(self, camera, bandWidthInStdDevs = 3):
        self.colourHuesDegrees = []
        self.colourHeusBandDegrees = []

        print "Calibrate the",self.numberOfColours,"colours.  Click on colour samples, press space when done for each colour."

        def mouseCallbackGetColour(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                colourValue = hsvFrame[y, x]
                print colourValue
                colourSamples.append(colourValue[0])

        for i in range(self.numberOfColours):
            colourSamples = []
            while True:
                # Get undistorted frame
                undistortedFrame = camera.getUndistortedFrame()

                # Convert to HSV
                hsvFrame = cv.cvtColor(undistortedFrame, cv.COLOR_BGR2HSV_FULL)

                windowTitle = "Calibrate Colour #"+str(i)
                cv.imshow(windowTitle, undistortedFrame)
                cv.setMouseCallback(windowTitle, mouseCallbackGetColour)

                key = cv.waitKey(1)
                if key == 1048608:  # space
                    # Calculate mean and standard deviation of colour samples
                    meanColourDegrees, sdColourDegrees = self.getMeanAndStdDevFromColourSamples(colourSamples)
                    print "mean colour (degrees): ",meanColourDegrees, "sd:", sdColourDegrees

                    self.colourHuesDegrees.append(meanColourDegrees) # convert to degrees
                    self.colourHeusBandDegrees.append(sdColourDegrees * bandWidthInStdDevs)
                    cv.destroyWindow(windowTitle)
                    break

    def displayCalibration(self):
        print "colourHuesDegrees: ", self.colourHuesDegrees
        print "colourHeusBandDegrees: ", self.colourHeusBandDegrees

    def setCalibration(self, colourHuesDegrees, colourHeusBandDegrees):
        self.colourHuesDegrees = colourHuesDegrees
        self.colourHeusBandDegrees = colourHeusBandDegrees

    def findColouredPixels(self, hsvFrame, hueDegrees, hueBandDegrees=20):
        if hueDegrees - hueBandDegrees > 0 and hueDegrees + hueBandDegrees < 360:
            # The band does not wrap around 0/360 degrees
            hueLowerBoundInt = np.uint8((float(hueDegrees - hueBandDegrees) / 360) * 255)
            hueUpperBoundInt = np.uint8((float(hueDegrees + hueBandDegrees) / 360) * 255)
            lowerBound = np.array([hueLowerBoundInt, 50, 50])
            upperBound = np.array([hueUpperBoundInt, 255, 255])
            mask = cv.inRange(hsvFrame, lowerBound, upperBound)
            # maskedFrame = cv.bitwise_and(frame, frame, mask=mask)
            return mask

        else:
            # The band wraps around zero, so deal with it in two parts

            # First deal with the portion of the band near zero degrees
            hueZeroBoundInt = np.uint8((float((hueDegrees + hueBandDegrees)%360) / 360) * 255)
            zero = np.array([0, 50, 50])
            zeroBound = np.array([hueZeroBoundInt, 255, 255])
            maskZero = cv.inRange(hsvFrame, zero, zeroBound)

            # Next, deal with the portion of the band zero 360 degrees
            hue360BoundInt = np.uint8((float((hueDegrees - hueBandDegrees)%360) / 360) * 255)
            limit360Bound = np.array([hue360BoundInt, 50, 50])
            limit360 = np.array([255, 255, 255])
            mask360 = cv.inRange(hsvFrame, limit360Bound, limit360)

            # The final mask is made of both the near-zero and near-360 masks
            mask = cv.bitwise_or(maskZero, mask360)
            return mask

    def findColouredBlobs(self, frame, debug = False):
        # Convert to HSV
        # global hsvFrame
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)  #

        masks = []
        self.keypoints = []
        for i in range(self.numberOfColours):
            masks.append(self.findColouredPixels(hsvFrame, self.colourHuesDegrees[i], self.colourHeusBandDegrees[i]))
            self.keypoints.append(self.detector.detect(masks[i]))

        if debug:
            frameWithKeypoints = frame
            for i in range(self.numberOfColours):
                # Show mask
                maskWindowName = "Mask #"+str(i)
                cv.imshow(maskWindowName, masks[i])
                # cv.moveWindow(maskWindowName, 0, 0)

                # Show keypoints
                KeyPointColour = [255 * cl for cl in colorsys.hsv_to_rgb(float(self.colourHuesDegrees[i]) / 360, 1, 1)]
                KeyPointColour.reverse()
                KeyPointColour = tuple(KeyPointColour)

                frameWithKeypoints = cv.drawKeypoints(frameWithKeypoints, self.keypoints[i], np.array([]), KeyPointColour,
                                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.imshow("Key Points", frameWithKeypoints)


    def findRobot(self, robot, camera, debug=False):
        '''Find robot assuming an unknown current location.  Scan the whole field.'''
        # Go through all the possible keypoint pairs.  Expected distance between the two colour patches is 69mm
        # (each of the squares has a side length of 49mm).  Rank all the possibilities based on how far away they
        # are from this ideal distance.  Select the pair that is the closest.
        distanceBetweenMarkers = 69 #mm
        colour1Keypoints = self.keypoints[robot.colours[0]]
        colour2Keypoints = self.keypoints[robot.colours[1]]

        minimumError = None # start with undefined error
        minimumErrorKeypoints = None
        for keypoint1 in colour1Keypoints:
            for keypoint2 in colour2Keypoints:
                # Calculate distance between the two keypoints
                distance = math.sqrt( (keypoint1.pt[0] - keypoint2.pt[0])**2 + \
                                      (keypoint1.pt[1] - keypoint2.pt[1])**2 )

                # Convert distance in pixels to real-world distance
                distance /= camera.pixelsPerMillimeter

                error = abs(distance - distanceBetweenMarkers)
                if minimumError is None or error < minimumError:
                    minimumError = error
                    minimumErrorKeypoints = (keypoint1, keypoint2)

        # If the minimum error is too great, then we haven't found the robot
        maximumAcceptableError = distanceBetweenMarkers # this says that we've got a 100% error
        if minimumError is None or minimumError > maximumAcceptableError:
            # We've failed to find the robot
            robot.location = None
            robot.orientation = None
            if debug:
                print "Failed to find robot:", robot
            return

        # print "minimum error", minimumError

        # Calculate the centre of the robot given the selected keypoints
        keypoint1, keypoint2 = minimumErrorKeypoints
        robot.location = (int((keypoint1.pt[0] + keypoint2.pt[0]) / 2), int((keypoint1.pt[1] + keypoint2.pt[1]) / 2))

        # Calculate orientation of robot given selected keypoints
        deltaX = keypoint1.pt[0] - keypoint2.pt[0]
        deltaY = keypoint1.pt[1] - keypoint2.pt[1]
        robot.orientation = math.degrees(math.atan2(deltaX, deltaY))

        # Debugging if requested
        if debug:
            print "robot:",robot

    def updateLocations(self, frame, camera, debug=False):
        '''Update the robots' locations.'''
        self.findColouredBlobs(frame, debug)

        for robot in self.robots:
            #if robot.location is None:
            self.findRobot(robot, camera, debug)

            if True:
                # Add circles around our robots
                cv.circle(frame, robot.location, 20, (0,0,0), 2)

                # Add a line to indicate the orientation
                if robot.location is not None and robot.orientation is not None:
                    cv.line(frame, robot.location, (int(robot.location[0] + 20*math.sin(math.radians(robot.orientation))), int(robot.location[1] + 20*math.cos(math.radians(robot.orientation)))), (0,0,0), 3)


def getMeanAndStdDevFromColourSamples(colourSamples):
    '''From a list of colourSamples, find the mean and standard deviation using a delta from the first sampled
    colour'''
    assert len(colourSamples) > 0

    # Source colour is the first of the colour samples
    sourceColour = int(colourSamples[0])

    # Go through the samples and delta between the current sample and the source.  There are two deltas, one in each
    # direction (possibly looping over 360 degrees).
    deltaUpColourSamples = [(int(x) - sourceColour) % 256 for x in colourSamples]
    deltaDownColourSamples = [-((sourceColour - (int(x))) % 256) for x in colourSamples]

    # Find the minimum of the two deltas
    deltaColourSamples = []
    for i in xrange(len(deltaDownColourSamples)):
        if abs(deltaDownColourSamples[i]) <= abs(deltaUpColourSamples[i]):
            deltaColourSamples.append(deltaDownColourSamples[i])
        else:
            deltaColourSamples.append(deltaUpColourSamples[i])

    # Find the mean and standard deviation of the (minimum) deltas
    meanDelta = float(sum(deltaColourSamples)) / len(deltaColourSamples)

    # Calculate the mean colour.
    meanColour = (sourceColour + meanDelta) % 256

    sdDelta = math.sqrt(
        float(sum([(x - meanDelta) ** 2 for x in deltaColourSamples])) / len(deltaColourSamples))

    # Convert the colours (from uint8) to degrees
    meanColourDegrees = float(meanColour) / 255 * 360
    sdColourDegrees = float(sdDelta) / 255 * 360

    return (meanColourDegrees, sdColourDegrees)


class BallDetector:
    def __init__(self, numberOfBalls):
        self.numberOfBalls = numberOfBalls

        # Setup SimpleBlobDetector parameters.
        def createDetectorParameters():
            params = cv.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 10
            params.maxThreshold = 200

            # Filter by Colour
            params.filterByColor = True
            params.blobColor = 255

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 150

            # Filter by Circularity
            params.filterByCircularity = False
            params.minCircularity = 0.1

            # Filter by Convexity
            params.filterByConvexity = False
            params.minConvexity = 0.87

            # Filter by Inertia
            params.filterByInertia = False
            params.minInertiaRatio = 0.01

            return params

        # Create SimpleBlobDetector (assumes OpenCV version 3)
        params = createDetectorParameters()
        self.detector = cv.SimpleBlobDetector_create(params)

        # To start, we do not know what the balls' colour is
        self.colourHueDegrees = None
        self.colourHeusBandDegrees = None

        # To start, we do not know where any of the balls are
        class Ball:
            def __init__(self):
                self.location = None

            def __repr__(self):
                return "("+str(self.location[0])+","+str(self.location[1])+")"

        self.balls = []
        for i in range(self.numberOfBalls):
            self.balls.append( Ball() )




    def calibrateColour(self, camera, bandWidthInStdDevs=3):
        self.colourHuesDegrees = []
        self.colourHeusBandDegrees = []

        print "Calibrate the balls' colour.  Click on colour samples, press space when done."

        def mouseCallbackGetColour(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                colourValue = hsvFrame[y, x]
                print colourValue
                colourSamples.append(colourValue[0])

        colourSamples = []
        while True:
            # Get undistorted frame
            undistortedFrame = camera.getUndistortedFrame()

            # Convert to HSV
            hsvFrame = cv.cvtColor(undistortedFrame, cv.COLOR_BGR2HSV_FULL)

            windowTitle = "Calibrate Ball Colour"
            cv.imshow(windowTitle, undistortedFrame)
            cv.setMouseCallback(windowTitle, mouseCallbackGetColour)

            key = cv.waitKey(1)
            if key == 1048608:  # space
                # Calculate mean and standard deviation of colour samples
                meanColourDegrees, sdColourDegrees = getMeanAndStdDevFromColourSamples(colourSamples)
                print "mean colour (degrees): ", meanColourDegrees, "sd:", sdColourDegrees

                self.colourHueDegrees = meanColourDegrees
                self.colourHeuBandDegrees = sdColourDegrees * bandWidthInStdDevs
                cv.destroyWindow(windowTitle)
                break

    def calibrateInitialLocations(self, camera):
        print "Calibrate the balls' initial positions.  Click on the centres of the balls."

        def mouseCallbackGetColour(event, x, y, flags, param):
            if event == cv.EVENT_LBUTTONDOWN:
                ballLocation = (x,y)
                print "ball location:", ballLocation
                ballLocations.append(ballLocation)

        ballLocations = []
        while True:
            # Get undistorted frame
            undistortedFrame = camera.getUndistortedFrame()

            windowTitle = "Calibrate Initial Ball Positions"
            cv.imshow(windowTitle, undistortedFrame)
            cv.setMouseCallback(windowTitle, mouseCallbackGetColour)

            key = cv.waitKey(1)
            if len(ballLocations) >= self.numberOfBalls:
                break

        for i in range(self.numberOfBalls):
            self.balls[i].location = ballLocations[i]

        pass

    def displayCalibration(self):
        print "colourHueDegrees: ", self.colourHueDegrees
        print "colourHeuBandDegrees: ", self.colourHeuBandDegrees

    def setCalibration(self, colourHueDegrees, colourHeuBandDegrees):
        self.colourHueDegrees = colourHueDegrees
        self.colourHeuBandDegrees = colourHeuBandDegrees

    def findColouredPixels(self, hsvFrame, hueDegrees, hueBandDegrees=20):
        if hueDegrees - hueBandDegrees > 0 and hueDegrees + hueBandDegrees < 360:
            # The band does not wrap around 0/360 degrees
            hueLowerBoundInt = np.uint8((float(hueDegrees - hueBandDegrees) / 360) * 255)
            hueUpperBoundInt = np.uint8((float(hueDegrees + hueBandDegrees) / 360) * 255)
            lowerBound = np.array([hueLowerBoundInt, 50, 50])
            upperBound = np.array([hueUpperBoundInt, 255, 255])
            mask = cv.inRange(hsvFrame, lowerBound, upperBound)
            # maskedFrame = cv.bitwise_and(frame, frame, mask=mask)
            return mask

        else:
            # The band wraps around zero, so deal with it in two parts

            # First deal with the portion of the band near zero degrees
            hueZeroBoundInt = np.uint8((float((hueDegrees + hueBandDegrees) % 360) / 360) * 255)
            zero = np.array([0, 50, 50])
            zeroBound = np.array([hueZeroBoundInt, 255, 255])
            maskZero = cv.inRange(hsvFrame, zero, zeroBound)

            # Next, deal with the portion of the band zero 360 degrees
            hue360BoundInt = np.uint8((float((hueDegrees - hueBandDegrees) % 360) / 360) * 255)
            limit360Bound = np.array([hue360BoundInt, 50, 50])
            limit360 = np.array([255, 255, 255])
            mask360 = cv.inRange(hsvFrame, limit360Bound, limit360)

            # The final mask is made of both the near-zero and near-360 masks
            mask = cv.bitwise_or(maskZero, mask360)
            return mask

    def findColouredBlobs(self, frame, debug=False):
        # Convert to HSV
        # global hsvFrame
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)  #

        mask = self.findColouredPixels(hsvFrame, self.colourHueDegrees, self.colourHeuBandDegrees)

        self.keypoints = self.detector.detect(mask)

        if debug:
            frameWithKeypoints = frame
            # Show mask
            maskWindowName = "Mask"
            cv.imshow(maskWindowName, mask)
            # cv.moveWindow(maskWindowName, 0, 0)

            # Show keypoints
            KeyPointColour = [255 * cl for cl in
                              colorsys.hsv_to_rgb(float(self.colourHueDegrees) / 360, 1, 1)]
            KeyPointColour.reverse()
            KeyPointColour = tuple(KeyPointColour)

            frameWithKeypoints = cv.drawKeypoints(frameWithKeypoints, self.keypoints, np.array([]),
                                                  KeyPointColour,
                                                  cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv.imshow("Key Points", frameWithKeypoints)

    def findBall(self, ball, camera, debug=False):
        pass

    def findBallWithKnownLocation(self, ball, camera, debug=False):
        if debug:
            print "looking for ball with previous location of", ball.location
        # If the minimum delta is too great, then we haven't found the ball
        self.maxDelta = 50 # mm per frame

        ball.minimumDelta = None

        for keypoint in self.keypoints:
            # Go through each ball and find the distance between this keypoint and the previous known location
            # Calculate distance between the keypoint and the ball
            distance = math.sqrt((keypoint.pt[0] - ball.location[0]) ** 2 + \
                                 (keypoint.pt[1] - ball.location[1]) ** 2)

            # Convert distance in pixels to real-world distance
            distance /= camera.pixelsPerMillimeter

            delta = abs(distance)
            if ball.minimumDelta is None or delta < ball.minimumDelta:
                if delta < self.maxDelta:
                    ball.minimumDelta = delta
                    ball.minimumDeltaLocation = keypoint.pt

        if ball.minimumDelta is None:
            # We've failed to find the ball
            if debug:
                print "Failed to find ball at previous location:", ball.location
            return
        else:
            # We've found the ball, update its current position
            if debug:
                print "Found ball at previous location:", ball.location, "change in position of", ball.minimumDelta, "new position:", ball.minimumDeltaLocation
            ball.location = ball.minimumDeltaLocation





    def updateLocations(self, frame, camera, debug=False):
        '''Update the balls' locations.'''
        self.findColouredBlobs(frame, debug)

        for ball in self.balls:
            if ball.location is None:
                print "The balls' initial locations should be specified and not found automatically."
                self.findBall(ball, camera, debug)
            else:
                self.findBallWithKnownLocation(ball, camera, debug)

            if True:
                # Add circles around our balls
                location = (int(ball.location[0]), int(ball.location[1]))
                cv.circle(frame, location, 20, (50, 50, 50), 2)


if __name__ == "__main__":
    # TEST BALL DETECTION
    fishEyeCamera = FishEyeCamera(0)
    ballDetector = BallDetector(4)
    robotDetector = RobotDetector(numberOfColours=2)
    #ballDetector.calibrateColour(fishEyeCamera)
    ballDetector.setCalibration(31.7, 2.6*3)
    ballDetector.calibrateInitialLocations(fishEyeCamera)
    robotDetector.setCalibration([140.5, 220.9], [12.5, 8.6])

    while True:
        undistortedFrame = fishEyeCamera.getUndistortedFrame()
        ballDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)
        robotDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)
        cv.imshow("Processed Frame", undistortedFrame)

        wk = cv.waitKey(1)
        if wk != -1:
            print wk
        if wk == 1048689: # q
            break

    cv.destroyAllWindows()
    exit()


    # TEST BALL DETECTION
    fishEyeCamera = FishEyeCamera(1)
    ballDetector = BallDetector(4)
    #ballDetector.calibrateColour(fishEyeCamera)
    ballDetector.setCalibration(31.7, 2.6*3)
    ballDetector.calibrateInitialLocations(fishEyeCamera)

    while True:
        undistortedFrame = fishEyeCamera.getUndistortedFrame()
        ballDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=True)
        #ballDetector.findColouredBlobs(undistortedFrame, debug=True)
        cv.imshow("Processed Frame", undistortedFrame)

        wk = cv.waitKey(1)
        if wk != -1:
            print wk
        if wk == 1048689: # q
            break

    cv.destroyAllWindows()
    exit()


    # TEST ROBOT DETECTION
    fishEyeCamera = FishEyeCamera(0)
    robotDetector = RobotDetector(numberOfColours=2)
    # robotDetector.calibrateColour(fishEyeCamera)
    # robotDetector.setCalibration([360.0, 140.5, 220.9], [15.7, 12.5, 8.6])
    robotDetector.setCalibration([140.5, 220.9], [12.5, 8.6])
    robotDetector.displayCalibration()

    # m1, sd1 = robotDetector.getMeanAndStdDevFromColourSamples([2,1,252])
    # m2, sd2 = robotDetector.getMeanAndStdDevFromColourSamples([252,2,1])
    # assert(m1==m2)
    #
    # print "sd1:", sd1, "sd2:",sd2
    # assert(sd1 == sd2)
    # print "mean:", m, "sd:",sd
    # cv.destroyAllWindows()
    # exit()


    while True:
        undistortedFrame = fishEyeCamera.getUndistortedFrame()
        cv.setMouseCallback("Undistorted Camera", draw_circle)

        robotDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)
        cv.imshow("Processed Frame", undistortedFrame)


        # Convert to HSV
        # hsvFrame = cv.cvtColor(undistortedFrame,cv.COLOR_BGR2HSV_FULL) #

        # robotDetector.findColouredBlobs(undistortedFrame, True)



        wk = cv.waitKey(1)
        if wk != -1:
            print wk
        if wk == 1048689: # q
            break

    cv.destroyAllWindows()
