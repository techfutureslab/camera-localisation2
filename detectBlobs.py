import cv2 as cv
import numpy as np
import time
import math

cam = cv.VideoCapture(0)
cam.set(3,1284)
cam.set(4,720)

robotSize = 50

# Camera calibration parameters obtained 2016-08-11 (by Mohsen and Matthew) using captureImages.py and calibrate.py
camera_matrix = np.array([[416.25456303, 0., 663.64459394], [0., 387.2264034, 380.49903696], [ 0., 0., 1.]])
dist_coefs= np.array([  2.00198060e-01,  -2.28216265e-01,   2.26068631e-04, -5.00177586e-04,   5.84619782e-02])
ret, frame = cam.read()
h, w = frame.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

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
params.minArea = 200

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
detector = cv.SimpleBlobDetector_create(params)

while True:
    startTime = time.time()
    ret, frame = cam.read()
    frame = cv.flip(cv.flip(frame, 1), -1)
    cv.imshow("Original", frame)
    frame = cv.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)

    # Convert to HSV
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL) #

    lb = np.array([156, 50,50])
    ub = np.array([184,255,255])
    maskBlue = cv.inRange(hsv_frame,lb, ub)
    blu = cv.bitwise_and(frame,frame,mask=maskBlue)
    #
    # lrz = np.array([0, 50,50])
    # lr = np.array([14, 255,255])
    # ur = np.array([241,50,50])
    # urf = np.array([255, 255,255])
    lnr =  np.array([14, 0,0])
    unr = np.array([241,255,255])
    maskNotRed = cv.inRange(hsv_frame,lnr, unr)
    # maskNotRed = cv.bitwise_not(maskRed)
    red = cv.bitwise_and(frame,frame,mask=maskNotRed)

    lg = np.array([70, 50,50])
    ug = np.array([100,255,255])
    maskGreen = cv.inRange(hsv_frame,lg, ug)
    grn = cv.bitwise_and(frame,frame,mask=maskGreen)

    # cv.imshow("Logic Frames", (np.logical_and(hsv_frame[:,:,1]>110 , hsv_frame[:,:,1]<150)).astype('float'))
    cv.imshow("Red", red)
    cv.imshow("Green", grn)
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

    # # Calculate centre of robot
    # robotPositionPx = None
    # try:
    #     grnPointX = green_keypoints[0].pt[0]
    #     grnPointY = green_keypoints[0].pt[1]
    #
    #     redPointX = red_keypoints[0].pt[0]
    #     redPointY = red_keypoints[0].pt[1]
    #
    #     deltaY = redPointY - grnPointY
    #     deltaX = redPointX - grnPointX
    #     acceptable_margin = 100 * 0.514  # (0.514 px /mm as measured 2016-08-11)
    #     distanceBetweenMarkers = math.sqrt(deltaX**2 + deltaY**2)
    #
    #     if distanceBetweenMarkers>acceptable_margin:
    #         print "Unacceptably far apart:", distanceBetweenMarkers, "mm"
    #         continue
    #
    #     robotPositionPx = (int((redPointX + grnPointX)/2) , int((redPointY + grnPointY) / 2))
    #     # print
    #     # cv.rectangle(frame_with_keypoints, (robotPositionPx[0] - 5, robotPositionPx[1] - 5),
    #     #              (robotPositionPx[0] + 5, robotPositionPx[1] + 5), 2, 2)
    #
    #     theta = math.degrees(math.atan2(deltaX,deltaY))
    #     print "robotPositionPx", robotPositionPx, "Orientation is ", theta,  "delta x:", deltaX, "delta Y:", deltaY
    #
    # except Exception as e:
    #     print "Exception thrown"
    #     print e
    #
    # if robotPositionPx is not None:
    #     cv.rectangle(frame_with_keypoints, (int(robotPositionPx[0]) - robotSize, int(robotPositionPx[1]) - robotSize),
    #                  (int(robotPositionPx[0]) + robotSize, int(robotPositionPx[1]) + robotSize), 2, 2)

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
