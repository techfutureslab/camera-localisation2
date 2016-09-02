import cv2 as cv
import cameraLocalisation2D
import gridMaker

# Define a grid
class Grid:
    def returnNearestGridPoint(self):
        pass

    def addGridToFrame(self, frame, corners):
        # Go through each grid point and add a dot representing the centre of each grid
        gridDotColour = (0,0,255) # red
        for p in self.points:
            # Convert points from 0..1 to pixels


            cv.circle(frame, p, 2, gridDotColour)


class RectangularGrid(Grid):
    def __init__(self):
        self.points = [(0.33,0.33),(0.33,0.67),(0.67,0.33),(0.67,0.67)]


def findCornersFromBalls(balls):
    # Find centre of balls
    sumX = float(0)
    sumY = float(0)
    for ball in balls:
        sumX += ball.location[0]
        sumY += ball.location[1]

    centre = (sumX/len(balls), sumY/len(balls))

    # Find upper left corner
    for ball in balls:
        if ball.location[0] < centre[0] and ball.location[1] < centre[1]:
            upperLeftBall = ball
        elif ball.location[0] < centre[0] and ball.location[1] > centre[1]:
            lowerLeftBall = ball
        elif ball.location[0] > centre[0] and ball.location[1] < centre[1]:
            upperRightBall = ball
        elif ball.location[0] > centre[0] and ball.location[1] > centre[1]:
            lowerRightBall = ball
        else:
            raise Exception("How'd we get here?")

    return {"upperLeft": upperLeftBall.location,
            "lowerLeft": lowerLeftBall.location,
            "upperRight": upperRightBall.location,
            "lowerRight": lowerRightBall.location}



# Place the grid points on the frame

# Given the location of grid points and the location of a robot, return which grid point is nearest


if __name__ == "__main__":
    fishEyeCamera = cameraLocalisation2D.FishEyeCamera(deviceID=0)

    ballDetector = cameraLocalisation2D.BallDetector(4)
    ballDetector.setCalibration(31.7, 2.6*3)
    ballDetector.calibrateInitialLocations(fishEyeCamera)



    robotDetector = cameraLocalisation2D.RobotDetector(numberOfColours=2)
    # robotDetector.calibrateColour(fishEyeCamera)
    # robotDetector.setCalibration([360.0, 140.5, 220.9], [15.7, 12.5, 8.6])
    robotDetector.setCalibration([140.5, 220.9], [12.5, 8.6])

    while True:
        undistortedFrame = fishEyeCamera.getUndistortedFrame()
        ballDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)
        robotDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=False)

        # Work out which balls represent which corner
        corners = findCornersFromBalls(ballDetector.balls)

        # Get corner points in order required by gridMaker
        gridMakerCorners = [(int(corners["upperLeft"][0]), int(corners["upperLeft"][1])),
                            (int(corners["upperRight"][0]), int(corners["upperRight"][1])),
                            (int(corners["lowerLeft"][0]), int(corners["lowerLeft"][1])),
                            (int(corners["lowerRight"][0]), int(corners["lowerRight"][1]))]

        #rectGrid.addGridToFrame(undistortedFrame, corners)
        gridMaker.draw(undistortedFrame, gridMakerCorners, (7,5))


        cv.imshow("Processed Frame", undistortedFrame)
        key = cv.waitKey(16) # 60 frames/sec
        if key != -1:
            print(key)
        if key == 1048689:  # q
            break
    exit()
