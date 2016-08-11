import cv2 as cv
import numpy as np
import time

# Camera calibration parameters obtained 2016-08-11 (by Mohsen and Matthew) using captureImages.py and calibrate.py
camera_matrix = np.array([[416.25456303, 0., 663.64459394], [0., 387.2264034, 380.49903696], [ 0., 0., 1.]])
dist_coefs= np.array([  2.00198060e-01,  -2.28216265e-01,   2.26068631e-04, -5.00177586e-04,   5.84619782e-02])


cam = cv.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)

ret, frame = cam.read()

h, w = frame.shape[:2]
print "Start Capturing"
cntr = 0
while True:
    ret, frame = cam.read()
    frame = cv.flip(cv.flip(frame, 1), -1)
    cv.imshow("Original", frame)

    wk = cv.waitKey(1)
    if wk!=-1:
        print wk
    if wk == 1048689:
        break

    newcameramtx, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coefs, (w, h), 1, (w, h))

    dst = cv.undistort(frame, camera_matrix, dist_coefs, None, newcameramtx)
    cv.rectangle(dst, (1280/2-5,720/2-5),(1280/2+5,720/2+5),2,2)
    cv.imshow("Results",dst)

cv.destroyAllWindows()


