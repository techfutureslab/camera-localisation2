import cv2 as cv
import numpy as np
import time
import math
import colorsys


def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        # cv.circle(img,(x,y),100,(255,0,0),-1)
        print hsvFrame[y,x]


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

        # Create SimpleBlobDetector (assumes OpenCV version 3)
        self.detector = cv.SimpleBlobDetector_create(params)
        self.colourHuesDegrees = [hue for hue in range(0,255,255/self.numberOfColours)]
        self.colourHeusBandDegrees = [10]*numberOfColours


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
                    # Turn colour samples into delta values
                    assert len(colourSamples) > 0
                    sourceColour = int(colourSamples[0])
                    deltaDownColourSamples = [int(x)-sourceColour for x in colourSamples]
                    deltaUpColourSamples = [int(x)+255-sourceColour for x in colourSamples]
                    deltaColourSamples = []
                    for i in xrange(len(deltaDownColourSamples)):
                        if abs(deltaDownColourSamples[i])<=abs(deltaUpColourSamples[i]):
                            deltaColourSamples.append(deltaDownColourSamples[i])
                        else:
                            deltaColourSamples.append(deltaUpColourSamples[i])
                    meanDelta = sum(deltaColourSamples) / len(deltaColourSamples)
                    sdDelta = math.sqrt(float(sum([x**2 for x in deltaColourSamples]))/len(deltaColourSamples))
                    meanColour = (sourceColour + meanDelta) % 255

                    meanColourDegrees = float(meanColour) / 255 * 360
                    sdColourDegrees = float(sdDelta) / 255 * 360
                    print "mean colour (degrees): ",meanColourDegrees, "sd:", sdColourDegrees


                    self.colourHuesDegrees.append(meanColourDegrees) # convert to degrees
                    self.colourHeusBandDegrees.append(sdColourDegrees * bandWidthInStdDevs)
                    cv.destroyWindow(windowTitle)
                    break

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
        global hsvFrame
        hsvFrame = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)  #

        masks = []
        keypoints = []
        for i in range(self.numberOfColours):
            masks.append(self.findColouredPixels(hsvFrame, self.colourHuesDegrees[i], self.colourHeusBandDegrees[i]))
            keypoints.append(self.detector.detect(masks[i]))

        if debug:
            frameWithKeypoints = frame
            for i in range(self.numberOfColours):
                # Show mask
                maskWindowName = "Mask #"+str(i)
                cv.imshow(maskWindowName, masks[i])
                cv.moveWindow(maskWindowName, 0, 0)

                # Show keypoints
                KeyPointColour = [255 * cl for cl in colorsys.hsv_to_rgb(float(self.colourHuesDegrees[i]) / 360, 1, 1)]
                KeyPointColour.reverse()
                KeyPointColour = tuple(KeyPointColour)

                frameWithKeypoints = cv.drawKeypoints(frameWithKeypoints, keypoints[i], np.array([]), KeyPointColour,
                                                    cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            pageTitle = "Color code: "+str(self.colourHuesDegrees)
            cv.imshow(pageTitle,frame)
            cv.imshow("Key Points", frameWithKeypoints)

    def getLocation(self):
        # Calculate centre of robot
        robotPositionPx = None
        try:
            grnPointX = green_keypoints[0].pt[0]
            grnPointY = green_keypoints[0].pt[1]

            redPointX = red_keypoints[0].pt[0]
            redPointY = red_keypoints[0].pt[1]

            deltaY = redPointY - grnPointY
            deltaX = redPointX - grnPointX
            acceptable_margin = 100 * 0.514  # (0.514 px /mm as measured 2016-08-11)
            distanceBetweenMarkers = math.sqrt(deltaX ** 2 + deltaY ** 2)

            if distanceBetweenMarkers > acceptable_margin:
                print "Unacceptably far apart:", distanceBetweenMarkers, "mm"
                # continue

            robotPositionPx = (int((redPointX + grnPointX) / 2), int((redPointY + grnPointY) / 2))
            # print
            # cv.rectangle(frame_with_keypoints, (robotPositionPx[0] - 5, robotPositionPx[1] - 5),
            #              (robotPositionPx[0] + 5, robotPositionPx[1] + 5), 2, 2)

            theta = math.degrees(math.atan2(deltaX, deltaY))
            print "robotPositionPx", robotPositionPx, "Orientation is ", theta, "delta x:", deltaX, "delta Y:", deltaY

        except Exception as e:
            print "Exception thrown"
            print e


fishEyeCamera = FishEyeCamera(0)
robotDetector = RobotDetector(2)
robotDetector.calibrateColour(fishEyeCamera)

while True:
    undistortedFrame = fishEyeCamera.getUndistortedFrame()
    cv.imshow("A Name", undistortedFrame)
    cv.setMouseCallback("A Name", draw_circle)

    # Convert to HSV
    hsvFrame = cv.cvtColor(undistortedFrame,cv.COLOR_BGR2HSV_FULL) #

    robotDetector.findColouredBlobs(undistortedFrame, True)
    # green = robotDetecter.findColouredPixels(hsvFrame, 2, 10) # Green
    # cv.imshow("Green", green)

    wk = cv.waitKey(1)
    if wk != -1:
        print wk
    if wk == 1048689: # q
        break

cv.destroyAllWindows()
exit()

###########################################################################################################################




while True:
    startTime = time.time()
    ret, frame = cam.read()
    frame = cv.flip(cv.flip(frame, 1), -1)
    #cv.imshow("Original", frame)
    frame = cv.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)

    # Convert to HSV
    hsvFrame = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL) #

    robotDetecter = RobotDetector()
    robotDetecter.findColouredPixels(hsvFrame, 240, 20) # Blue

    key = waitKey(0)
    continue

    lb = np.array([156, 50,50])
    ub = np.array([184,255,255])
    maskBlue = cv.inRange(hsv_frame,lb, ub)
    blu = cv.bitwise_and(frame,frame,mask=maskBlue)

    lrz = np.array([0, 50,50])
    lr = np.array([14, 255,255])
    ur = np.array([241,50,50])
    urf = np.array([255, 255,255])
    maskRedLower = cv.inRange(hsv_frame,lrz, lr)
    maskRedUpper = cv.inRange(hsv_frame,ur, urf)
    maskRed = cv.bitwise_or(maskRedLower, maskRedUpper)
    red = cv.bitwise_and(frame,frame,mask=maskRed)

    lg = np.array([70, 50,50])
    ug = np.array([100,255,255])
    maskGreen = cv.inRange(hsv_frame,lg, ug)
    grn = cv.bitwise_and(frame,frame,mask=maskGreen)

    # cv.imshow("Logic Frames", (np.logical_and(hsv_frame[:,:,1]>110 , hsv_frame[:,:,1]<150)).astype('float'))
    #cv.imshow("Red", red)
    #cv.imshow("Green", grn)
    #cv.imshow("Blue", blu)

    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    # Detect red blobs
    red_keypoints = detector.detect(red)
    if len(red_keypoints) == 0:
        continue
    frame_with_keypoints = cv.drawKeypoints(frame, red_keypoints, np.array([]), (0, 0, 255),
                                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Detect green blobs
    green_keypoints = detector.detect(grn)
    if len(green_keypoints) ==0:
        continue
    frame_with_keypoints = cv.drawKeypoints(frame_with_keypoints, [green_keypoints[0]], np.array([]), (0, 255, 0),
                                            cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Calculate centre of robot
    robotPositionPx = None
    try:
        grnPointX = green_keypoints[0].pt[0]
        grnPointY = green_keypoints[0].pt[1]

        redPointX = red_keypoints[0].pt[0]
        redPointY = red_keypoints[0].pt[1]

        deltaY = redPointY - grnPointY
        deltaX = redPointX - grnPointX
        acceptable_margin = 100 * 0.514  # (0.514 px /mm as measured 2016-08-11)
        distanceBetweenMarkers = math.sqrt(deltaX**2 + deltaY**2)

        if distanceBetweenMarkers>acceptable_margin:
            print "Unacceptably far apart:", distanceBetweenMarkers, "mm"
            continue

        robotPositionPx = (int((redPointX + grnPointX)/2) , int((redPointY + grnPointY) / 2))
        # print
        # cv.rectangle(frame_with_keypoints, (robotPositionPx[0] - 5, robotPositionPx[1] - 5),
        #              (robotPositionPx[0] + 5, robotPositionPx[1] + 5), 2, 2)

        theta = math.degrees(math.atan2(deltaX,deltaY))
        print "robotPositionPx", robotPositionPx, "Orientation is ", theta,  "delta x:", deltaX, "delta Y:", deltaY

    except Exception as e:
        print "Exception thrown"
        print e

    robotSize = 50

    if robotPositionPx is not None:
        cv.rectangle(frame_with_keypoints, (int(robotPositionPx[0]) - robotSize, int(robotPositionPx[1]) - robotSize),
                     (int(robotPositionPx[0]) + robotSize, int(robotPositionPx[1]) + robotSize), 2, 2)

    cv.imshow("Robot Location", frame_with_keypoints)

    endTime = time.time()
    print "frames/sec: ", 1/(endTime-startTime)

    wk = cv.waitKey(1)
    if wk != -1:
        print wk
        if wk == 1048608:
            print("hsv:", hsv_frame[320, 240,0])
            print("rgb:", frame[320, 240])
    if wk == 1048689:
        break

cv.destroyAllWindows()
