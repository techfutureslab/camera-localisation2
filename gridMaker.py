import cv2
import numpy as np



def findLinePoints(mPoint,segs=(3,4)):
    # output = Matrix of [NxM]
    uN = []  # upper N
    lN = []  # lower N
    lM = []  # left M
    rM = []  # right M
    oP = [] # output
    # upperN
    DeltaX = float(mPoint[1][0] - mPoint[0][0])
    DeltaY = float(mPoint[1][1] - mPoint[0][1])
    for n in range(1,segs[0]):
        uN.append((int(mPoint[0][0]+(n*DeltaX)/(segs[0])),int(mPoint[0][1]+(n*DeltaY)/(segs[0]))))
        oP.append((int(mPoint[0][0]+(n*DeltaX)/(segs[0])),int(mPoint[0][1]+(n*DeltaY)/(segs[0]))))
    # lowerN
    DeltaX = float(mPoint[3][0] - mPoint[2][0])
    DeltaY = float(mPoint[3][1] - mPoint[2][1])
    for n in range(1,segs[0]):
        lN.append((int(mPoint[2][0]+(n*DeltaX)/(segs[0])),int(mPoint[2][1]+(n*DeltaY)/(segs[0]))))
        oP.append((int(mPoint[2][0]+(n*DeltaX)/(segs[0])),int(mPoint[2][1]+(n*DeltaY)/(segs[0]))))
    # leftM
    DeltaX = float(mPoint[2][0] - mPoint[0][0])
    DeltaY = float(mPoint[2][1] - mPoint[0][1])
    for n in range(1,segs[1]):
        lM.append((int(mPoint[0][0]+(n*DeltaX)/(segs[1])),int(mPoint[0][1]+(n*DeltaY)/(segs[1]))))
        oP.append((int(mPoint[0][0]+(n*DeltaX)/(segs[1])),int(mPoint[0][1]+(n*DeltaY)/(segs[1]))))
    # rightM
    DeltaX = float(mPoint[3][0] - mPoint[1][0])
    DeltaY = float(mPoint[3][1] - mPoint[1][1])
    for n in range(1,segs[1]):
        rM.append((int(mPoint[1][0]+(n*DeltaX)/(segs[1])),int(mPoint[1][1]+(n*DeltaY)/(segs[1]))))
        oP.append((int(mPoint[1][0]+(n*DeltaX)/(segs[1])),int(mPoint[1][1]+(n*DeltaY)/(segs[1]))))
    return oP, uN,lN,lM,rM # upper N points
    pass


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return int(x), int(y)


def findMiddlePoints(uN,lN,lM,rM):

    middlePoints = []
    # B = np.array([[0],[0]])
    # A = np.array([[0, 1], [0, 1]])
    assert len(uN) == len(lN)
    assert len(lM) == len(rM)

    for n in range(len(uN)):
        for m in range(len(lM)):
            middlePoints.append(line_intersection((uN[n],lN[n]),(rM[m],lM[m])))
    return middlePoints


def pixel2Board(point):
    #  TODO
    res = []

    return res



def draw(img, mPoint, segments=(3,4)):
    if len(mPoint) == 4:
        cv2.line(img, mPoint[0], mPoint[1], (0, 250, 0), 3)
        cv2.line(img, mPoint[2], mPoint[3], (0, 250, 0), 3)
        cv2.line(img, mPoint[0], mPoint[2], (0, 0, 250), 3)
        cv2.line(img, mPoint[1], mPoint[3], (0, 0, 250), 3)
        points, uN, lN, lM, rM = findLinePoints(mPoint, segments)
        for p in range(len(points)):
            cv2.circle(img, points[p], 15, (0, 255, 255), -1)
        for n in range(segments[0] - 1):
            cv2.line(img, uN[n], lN[n], (0, 155, 155), 2)
        for m in range(segments[1] - 1):
            cv2.line(img, rM[m], lM[m], (0, 155, 155), 2)
        middlePoints = findMiddlePoints(uN, lN, lM, rM)
        for p in range(len(middlePoints)):
            cv2.circle(img, middlePoints[p], 12, (155, 0, 0), -1)


if __name__ == "__main__":
    segments = (5, 7)
    mPoint = []

    # mouse callback function
    def draw_circle(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            cv2.circle(img, (x, y), 20, (255, 0, 0), -1)
            mPoint.append((x, y))


    # Create a black image, a window and bind the function to window
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_circle)

    while(1):
        cv2.imshow('image',img)
        draw(img, mPoint)


        if cv2.waitKey(20) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print(mPoint)