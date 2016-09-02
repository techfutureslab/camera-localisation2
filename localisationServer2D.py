import cameraLocalisation2D
import socketserver
import cv2 as cv

class Handler(socketserver.BaseRequestHandler):
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
            # TODO: It would be much better if the camera and detector system were
            # not part of this handler
            undistortedFrame = self.fishEyeCamera.getUndistortedFrame()
            self.robotDetector.updateLocations(undistortedFrame, self.fishEyeCamera, debug=False)
            cv.imshow("Processed Frame", undistortedFrame)

            result = ""
            for robot in self.robotDetector.robots:
                result += "\n" + str(robot)
            self.request.sendall(result)

            key = cv.waitKey(16)  # 60 frames/sec
            if key != -1:
                print(key)
            if key == 1048689:  # q
                break

if __name__ == "__main__":
    # fishEyeCamera = cameraLocalisation2D.FishEyeCamera(deviceID=0)
    # robotDetector = cameraLocalisation2D.RobotDetector(numberOfColours=3)
    # while True:
    #     undistortedFrame = fishEyeCamera.getUndistortedFrame()
    #     robotDetector.updateLocations(undistortedFrame, fishEyeCamera, debug=True)
    #     cv.imshow("Processed Frame", undistortedFrame)
    #     key = cv.waitKey(16) # 60 frames/sec
    #     if key != -1:
    #         print key
    #     if key == 1048689:  # q
    #         break
    # exit()


    host = "localhost"
    port = 9999

    server = socketserver.TCPServer((host, port), Handler)

    print("Starting Server")
    server.serve_forever()
    print("Server Finished")
