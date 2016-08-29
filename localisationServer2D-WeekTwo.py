import cameraLocalisation2D
import SocketServer
import cv2 as cv

class Handler(SocketServer.BaseRequestHandler):
    def setup(self):
        self.fishEyeCamera = cameraLocalisation2D.FishEyeCamera(deviceID=1)
        self.robotDetector = cameraLocalisation2D.RobotDetector(numberOfColours=3)
        # robotDetector.calibrateColour(self.fishEyeCamera)
        self.robotDetector.setCalibration([360.0, 140.5, 220.9], [15.7, 12.5, 8.6])
        self.robotDetector.displayCalibration()

    def handle(self):
        # self.data = self.request.recv(1024).strip()
        # print "{} wrote".format(self.client_address[0])
        # print self.data

        while True:
            undistortedFrame = self.fishEyeCamera.getUndistortedFrame()
            self.robotDetector.updateLocations(undistortedFrame, self.fishEyeCamera, debug=False)
            cv.imshow("Processed Frame", undistortedFrame)

            result = ""
            for robot in self.robotDetector.robots:
                result += "\n" + str(robot)
            self.request.sendall(result)

if __name__ == "__main__":
    host = "localhost"
    port = 9999

    server = SocketServer.TCPServer((host, port), Handler)

    server.serve_forever()
