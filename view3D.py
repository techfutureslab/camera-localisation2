import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)
camSide = cv2.VideoCapture(1)
# cam.set(3,1280)
# cam.set(4,720)
# camSide.set(3,1280)
# camSide.set(4,720)

# Get the width and height of the camera image
ret, frame = cam.read()
h, w = frame.shape[:2]

while True:
    ret, frame = cam.read()
    ret, frameSide = camSide.read()
    # ff_frame = cv2.flip(cv2.flip(frame,1),-1)
    cv2.imshow("Top Image", frame)
    cv2.imshow("Side Image", frameSide)

    key = cv2.waitKey(33)
    if key ==  1048689:  # q
        break
    elif key != -1:
        print key

cv2.destroyAllWindows()
cam.release()
