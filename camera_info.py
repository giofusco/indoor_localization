import numpy as np
from math import atan

IPHONE8_640x360 = "iPhone8_640x480"
IPHONE8_1280x720 = "iPhone8_1280x720"
CAMERA_MATRIX = "camera_matrix"
DIST_COEFFS = "dist_coeffs"


def get_camera_params(cameraID=IPHONE8_640x360):

    camera_params = {}

    if cameraID == IPHONE8_640x360:
        camera_params[CAMERA_MATRIX] = np.matrix([[496.76392161,  0., 179.4307037],
                                                                     [0., 492.06136884, 318.7074295],
                                                                       [0., 0., 1.]])
        camera_params[DIST_COEFFS] = np.array([2.79707782e-01, -1.51142752e+00, -1.93469830e-03,
                                                            -1.46837375e-03, 2.63094958e+00])

    if cameraID == IPHONE8_1280x720:
        camera_params[CAMERA_MATRIX] = np.matrix([[1.00744647e+03, 0.00000000e+00, 3.55952092e+02],
                                                                [0.00000000e+00, 9.98023688e+02, 6.24955395e+02],
                                                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

        camera_params[DIST_COEFFS] = np.array([3.18731405e-01, -2.25074134e+00, -8.40742307e-03,
                                                            -1.68504587e-03, 6.04493696e+00])

    return camera_params

def get_camera_angle_per_pixel(cameraID=IPHONE8_640x360):
    if cameraID == IPHONE8_640x360:
        w = 360/2
        fx = 496.76392161
        tan_alpha = w / fx
        a_p_p = atan(tan_alpha)/w

        return a_p_p