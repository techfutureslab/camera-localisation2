 # Fundamental matrix using 8 points
# [[ -9.01047758e-09   2.77077394e-06  -1.85608077e-03]
#  [  2.05331917e-06   7.27732510e-08  -1.36590441e-03]
#  [  2.80996733e-04  -1.75639180e-03   1.00000000e+00]]


import cameraLocalisation2D
import cv2
import numpy as np


F = np.array([[ -9.01047758e-09,   2.77077394e-06,  -1.85608077e-03], [  2.05331917e-06 ,  7.27732510e-08,  -1.36590441e-03] , [  2.80996733e-04 , -1.75639180e-03 ,  1.00000000e+00]])

Point = (774,310,1)
Point2 =(926,192,1)

def drawlines(img1, lines, pts1):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    # img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1  in zip(lines, pts1):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = list(map(int, [0, -r[2] / r[1]]))
        x1, y1 = list(map(int, [c, -(r[2] + r[0] * c) / r[1]]))
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
    return img1

def onMouseClicked(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # cv.circle(img,(x,y),100,(255,0,0),-1)
        # print hsvFrame[y,x]
        pass


def findTheOtherLine(F,Point):
    point = np.array([[Point[0]],[Point[1]],[Point[2]]])
    lineParam = np.dot(F,point)
    print(lineParam)
    return lineParam


if __name__ == "__main__":
    # TEST BALL DETECTION
    try:
        fishEyeCameraTop = cameraLocalisation2D.FishEyeCamera(1)
    except:
        print("Failed to correctly set up the top camera")
        exit()

    try:
        fishEyeCameraSide= cameraLocalisation2D.FishEyeCamera(2)
    except:
        print("Failed to correctly set up the side camera")
        exit()


    lineParam = findTheOtherLine(F,Point)
    lineParam2 = findTheOtherLine(F,Point2)

    while True:
        undistortedFrameTop = fishEyeCameraTop.getUndistortedFrame()
        undistortedFrameSide = fishEyeCameraSide.getUndistortedFrame()
        # undistortedFrameSide = drawlines(undistortedFrameSide,lineParam, (100,100))

        a = lineParam[0]
        b = lineParam[1]
        c = lineParam[2]

        A = lineParam2[0]
        B = lineParam2[1]
        C = lineParam2[2]
        undistortedFrameSide = cv2.line(undistortedFrameSide, (1, int(-(a+c)/b)), (1280, int(-(1280*a+c)/b)),(255,0,0))

        undistortedFrameSide = cv2.line(undistortedFrameSide, (1, int(-(A+C)/B)), (1280, int(-(1280*A+C)/B)),(0,255,0))

        cv2.imshow("Top View",undistortedFrameTop)
        cv2.imshow("Side View",undistortedFrameSide)


        # cv2.moveWindow()

        wk = cv2.waitKey(1)
        if wk != -1:
            print(wk)
        if wk == 1048689 or wk == 113:  # q
            break

    cv2.destroyAllWindows()

