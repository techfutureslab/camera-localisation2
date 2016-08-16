import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)

# Get the width and height of the camera image
ret, frame = cam.read()
h, w = frame.shape[:2]

while True:
    ret, frame = cam.read()
    # ff_frame = cv2.flip(cv2.flip(frame,1),-1)
    cv2.imshow("Raw Image", frame)

    key = cv2.waitKey(33)
    if key ==  1048689:  # q
        break
    elif key != -1:
        print key

cv2.destroyAllWindows()
cam.release()