import numpy as np
import cv2
import glob

cam = cv2.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)


boxUpperLeft = None
frame = None
movedY = None
movedX = None
rect = None

def mouseSelection(event,x,y,flags,param):
    global boxUpperLeft, movedX,movedY
    global frame
    if event == cv2.EVENT_LBUTTONDOWN:
        # cv.circle(img,(x,y),100,(255,0,0),-1)
        boxUpperLeft = (x,y)
        print "x:",x,"y: ",y
    elif event==cv2.EVENT_MOUSEMOVE:
        if boxUpperLeft is not None:
            print "Mouse moved"
            print x,y
            movedX = x
            movedY = y
    elif event ==cv2.EVENT_LBUTTONUP:
        rect = (boxUpperLeft[0],boxUpperLeft[1],x,y)
        # With the rectangle, collect each pixel's hue in an array
        height = abs(boxUpperLeft[1] - y)
        width = abs(boxUpperLeft[0] - x)
        minX = min(boxUpperLeft[0], x)
        minY = min(boxUpperLeft[1], y)
        croppedFrame = frame[minY:minY+height, minX:minX+width]


        # Reset the state
        boxUpperLeft = None
        movedY = None
        movedX = None






def returnSampleColour(rect,frame):
    if rect:
        selectedBox = frame[rect(0):rect(1)]
        sample = selectedBox[:]
        return sample


# Get the width and height of the camera image
def getFrameSize():
    ret, frame = cam.read()
    h, w = frame.shape[:2]
    return (h,w)

h,w = getFrameSize()

colourSample = []

while True:
    ret, frame = cam.read()
    # ff_frame = cv2.flip(cv2.flip(frame,1),-1)
    if movedY is not None and boxUpperLeft is not None:
        cv2.rectangle(frame, boxUpperLeft, (movedX, movedY), (0, 255, 0),1)
    cv2.imshow("image", frame)
    cv2.setMouseCallback('image', mouseSelection)
    colourSample.append(returnSampleColour())
    key = cv2.waitKey(33)
    if key ==  1048689:  # q
        break
    elif key != -1:
        print key

cv2.destroyAllWindows()
cam.release()
