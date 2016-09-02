import cv2 as cv
import numpy as np
import time
import math

cam = cv.VideoCapture(0)
cam.set(3,1284)
cam.set(4,720)

robotSize = 50


def mouseSelection(event,x,y,flags,param):
    global boxUpperLeft, movedX,movedY
    global frame
    if event == cv.EVENT_RBUTTONDBLCLK:
        # cv.circle(img,(x,y),100,(255,0,0),-1)
        boxUpperLeft = (x,y)
        print("x:",x,"y: ",y)
        # str = "BGR: {} /  HSV: {}".format( (frame[y,x,:]), (hsv[y,x,:]))
        print(str)

ret, frame = cam.read()
h, w = frame.shape[:2]
while True:
    startTime = time.time()
    ret, frame = cam.read()
    frame = cv.flip(cv.flip(frame, 1), -1)
    cv.imshow("Original", frame)
    cv.setMouseCallback('Original', mouseSelection)

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)
    wk = cv.waitKey(1)
    if wk != -1:
        print(wk)
        if wk == 1048608:
            # print("hsv:", hsv_frame[320, 240, 0])
            print(("rgb:", frame[320, 240]))
    if wk == 1048689:
        break

cv.destroyAllWindows()