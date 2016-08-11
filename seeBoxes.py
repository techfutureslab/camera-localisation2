import cv2 as cv
import numpy as np
import time


cam = cv.VideoCapture(0)

cam.set(3,1280)
cam.set(4,720)


while True:
    ret, frame = cam.read()
    hsv_frame = cv.cvtColor(frame,cv.COLOR_BGR2HSV_FULL) #
    cv.imshow("Color Frames", (frame))

    lb = np.array([156, 50,50])
    ub = np.array([184,255,255])
    maskBlue = cv.inRange(hsv_frame,lb, ub)
    blu = cv.bitwise_and(frame,frame,mask=maskBlue)

    lrz = np.array([0, 50,50])
    lr = np.array([14, 255,255])
    ur = np.array([241,50,50])
    urf = np.array([255, 255,255])
    maskRedLower = cv.inRange(hsv_frame,lrz, lr)
    maskRedUpper=  cv.inRange(hsv_frame,ur, urf)
    maskRed = cv.bitwise_or(maskRedLower, maskRedUpper)
    red = cv.bitwise_and(frame,frame,mask=maskRed)

    lg = np.array([70, 50,50])
    ug = np.array([100,255,255])
    maskGreen = cv.inRange(hsv_frame,lg, ug)
    grn = cv.bitwise_and(frame,frame,mask=maskGreen)

    # cv.imshow("Logic Frames", (np.logical_and(hsv_frame[:,:,1]>110 , hsv_frame[:,:,1]<150)).astype('float'))
    cv.imshow("Red", red)
    cv.imshow("Green", grn)
    cv.imshow("Blue", blu)
    wk = cv.waitKey(1)
    if wk != -1:
        print wk
        if wk == 1048608:
            print("hsv:", hsv_frame[320, 240,0])
            print("rgb:", frame[320, 240])
    if wk == 1048689:
        break


cv.imwrite("green.jpg", grn)
cv.destroyAllWindows()
