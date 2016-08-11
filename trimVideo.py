import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)

# Get the width and height of the camera image
ret, frame = cam.read()
h, w = frame.shape[:2]


upperLeftCorner = None
lowerRightCorner = None

# mouse callback function
def mouse_click(event,x,y,flags,param):
    global upperLeftCorner
    global lowerRightCorner
    if event == cv2.EVENT_LBUTTONDOWN:
        print "Mouse down at location ", x,y
        upperLeftCorner = (x,y)
    if event == cv2.EVENT_LBUTTONUP:
        print "Mouse up at location ", x,y
        lowerRightCorner = (x,y)

# Create a black image, a window and bind the function to window



while True:
    ret, frame = cam.read()
    frame = cv2.flip(cv2.flip(frame,1),-1)
    if upperLeftCorner is not None and lowerRightCorner is not None:
        newFrame = frame[upperLeftCorner[1]:lowerRightCorner[1], upperLeftCorner[0]:lowerRightCorner[0]]
    else:
        newFrame = frame
    cv2.imshow("Raw Image", newFrame)
    cv2.setMouseCallback('Raw Image', mouse_click)

    key = cv2.waitKey(33)
    if key ==  1048689:  # q
        break
    elif key != -1:
        print key

cv2.destroyAllWindows()
cam.release()