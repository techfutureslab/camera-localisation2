
import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)


ret, frame = cam.read()

h, w = frame.shape[:2]
print "Start Capturing"
cntr = 0
while True:
    ret, frame = cam.read()
    frame = cv2.flip(cv2.flip(frame,1),-1)
    cv2.imshow("Frames", frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 9), None)


    # If found, add object points, image points (after refining them)
    key = cv2.waitKey(33)
    if key ==  1048689:  # q
        break
    elif key == 1048608: # space
        if ret:
            # Capture image
            cntr += 1
            filename = "Image"+str(cntr)+".png"
            print "Writing", filename
            cv2.imwrite(filename,frame)
        else:
            print "Failed to find chessboard"
    elif key != -1:
        print key

cv2.destroyAllWindows()
cam.release()