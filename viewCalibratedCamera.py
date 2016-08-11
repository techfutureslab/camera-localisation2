import cv2 as cv
import numpy as np
import time


# RMS: 0.420890075555
# camera matrix:
#  [[ 545.41279617    0.          619.88013591]
#  [   0.          586.55407074  346.22337371]
#  [   0.            0.            1.        ]]
# distortion coefficients:  [ 0.64132761 -1.01698547 -0.0093332  -0.00630371  0.44539262]


#
# camera_matrix= np.array([[ 265.97941148,0.,308.316857  ], [   0.,246.14208127, 233.02025798], [   0.,0., 1.]])
# dist_coefs =  np.array([ 0.17784696, -0.14629628,  0.00539734, -0.01584541,  0.02885636])
#

# camera_matrix= np.array([[   1.60365490e+03,  0.00000000e+00,   6.10370626e+02],  [  0.00000000e+00,   1.57559977e+03,   3.44901648e+02], [  0.00000000e+00,   0.00000000e+00 ,  1.00000000e+00]])
# distortion_coefficients= np.array([  3.86197595e+00,  -6.89177553e+01,   2.98078644e-02,   2.97723905e-02,    3.32743150e+02])



# camera_matrix= np.array([[   545.41279617,  0.00000000e+00,   619.88013591],  [  0.00000000e+00,   586.55407074,   346.22337371], [  0.00000000e+00,   0.00000000e+00 ,  1.00000000e+00]])
# dist_coefs= np.array([ 0.64132761, -1.01698547, -0.0093332,  -0.00630371,  0.44539262])

# camera_matrix = np.array([[ 288.48089301,    0.        ,  337.48099838],       [   0.        ,  272.09781401,  221.38623909],       [   0.        ,    0.        ,    1.        ]])
# dist_coefs= np.array([ 0.27425983, -0.38484375, -0.00077867,  0.00181614,  0.14097943])

camera_matrix = np.array([[ 416.25456303,    0.        ,  663.64459394],       [   0.        ,  387.2264034 ,  380.49903696],       [   0.        ,    0.        ,    1.        ]])
dist_coefs= np.array([  2.00198060e-01,  -2.28216265e-01,   2.26068631e-04, -5.00177586e-04,   5.84619782e-02])


cam = cv.VideoCapture(0)
cam.set(3,1280)
cam.set(4,720)
# cam.set(10, 0)

ret, frame = cam.read()

h, w = frame.shape[:2]
# time.sleep(5)
print "Start Capturing"
#for i in range(1,14):
cntr = 0
while True:
    ret, frame = cam.read()
    frame = cv.flip(cv.flip(frame, 1), -1)
    cv.imshow("Original", frame)


    # gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    # ret, corners = cv.findChessboardCorners(gray, (7,6),None)

    # if ret:
    #     print "Success!"
    #     print "Writing file #", cntr
    #     cv.imwrite("ImagesSeries{}.png".format(cntr),frame)
    #     cntr += 1

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



# RMS= 0.886792275858
# camera_matrix= [[ 265.97941148,0.,308.316857  ], [   0.,246.14208127, 233.02025798], [   0.,0., 1.]]
# distortion_coefficients=  [ 0.17784696, -0.14629628,  0.00539734, -0.01584541,  0.02885636]

# RMS: 0.423503238749
# camera matrix:
#  [[  1.60365490e+03   0.00000000e+00   6.10370626e+02]
#  [  0.00000000e+00   1.57559977e+03   3.44901648e+02]
#  [  0.00000000e+00   0.00000000e+00   1.00000000e+00]]
# distortion coefficients:  [  3.86197595e+00  -6.89177553e+01   2.98078644e-02   2.97723905e-02
#    3.32743150e+02]

#
# RMS: 0.420890075555
# camera matrix:
#  [[ 545.41279617    0.          619.88013591]
#  [   0.          586.55407074  346.22337371]
#  [   0.            0.            1.        ]]
# distortion coefficients:  [ 0.64132761 -1.01698547 -0.0093332  -0.00630371  0.44539262]
