import input.data_constants as dc
import numpy as np
import cv2
from cv2 import aruco
import math
import camera_info

DETECTION_WINDOW_NAME = "Detection"

class MarkerDetector:

    def __init__(self, name, min_consecutive_frames=3, max_marker_distance_meter=3., marker_length_m=.159, camera_id=camera_info.IPHONE8_640x360):

        self.camera_calibration = camera_info.get_camera_params(camera_id)

       #iPhone 8 camera calibration 360
        self.camera_matrix = self.camera_calibration[camera_info.CAMERA_MATRIX]
        self.dist_coeffs = self.camera_calibration[camera_info.DIST_COEFFS]

        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        self.marker_length_m = marker_length_m  # in m
        self.aruco_params = aruco.DetectorParameters_create()
        self.aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.aruco_params.cornerRefinementWinSize = 3
        self.aruco_params.minMarkerPerimeterRate = 0.05
        self.aruco_params.maxErroneousBitsInBorderRate = 0.0
        self.detections = {}
        self.best_detection_id = None
        self.tmp_best_id = None
        self.name = name
        self.num_frame_id_detection = 0
        self.min_consecutive_frames = min_consecutive_frames
        self.max_marker_distance_meter = max_marker_distance_meter
        self.last_frame_RGB = None
        self.enabled = True

    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True



    def set_marker_size(self, size_m):
        self.marker_length_m = size_m

    def get_observations(self, annotated_map):
        marker_id = self.best_detection_id
        if marker_id is not None:
            tvec = self.detections[marker_id]['tvec']
            rvec = self.detections[marker_id]['rvec']
            if marker_id in annotated_map.map_landmarks_dict and tvec[2] <= self.max_marker_distance_meter:
                marker_position_XY, yaw_marker = \
                    MarkerDetector.compute_XY_position_on_marker_detection(marker_id, tvec, rvec, annotated_map)
                return marker_position_XY, yaw_marker
            else:
                return None, None
        else:
            return None, None

    def update(self, data):
        self.detections = {}

        if not data[dc.IMAGE] is None and self.enabled:
            self.last_frame_RGB = np.copy(data[dc.IMAGE])
            img_gray = cv2.cvtColor(data[dc.IMAGE], cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(img_gray, self.aruco_dict, parameters=self.aruco_params)
            # print ids

            # cv2.waitKey(-1)
            if ids is not None:
                biggest_id, idx, _ = self._find_biggest_detection(ids, corners)
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, self.marker_length_m, self.camera_matrix,
                                                                self.dist_coeffs)

                for c in range(0, len(ids)):
                    self.detections[str(ids[c][0])] = {}
                    self.detections[str(ids[c][0])]['tvec'] = tvec[c][0]
                    self.detections[str(ids[c][0])]['rvec'] = rvec[c][0]
                    self.detections[str(ids[c][0])]['corners'] = corners[c]

                if biggest_id == self.tmp_best_id:
                    self.num_frame_id_detection += 1
                    self.best_detection_id = None
                else:
                    self.tmp_best_id = biggest_id
                    self.num_frame_id_detection = 1
                    self.best_detection_id = None
                if self.num_frame_id_detection >= self.min_consecutive_frames:
                    self.best_detection_id = self.tmp_best_id
            else:
                self.best_detection_id = None
                self.num_frame_id_detection = 0
            # if self.best_detection_id is not None:
                # print(self.best_detection_id)
                # self.plot_detection(self.best_detection_id)

    def _get_marker_height(self, corners):
        minY = 1e6
        maxY = -1
        minX = 1e6
        maxX = -1
        for c in range(0, 4):
            if corners[0][c][1] > maxY:
                maxY = corners[0][c][1]
            if corners[0][c][1] < minY:
                minY = corners[0][c][1]
            if corners[0][c][0] > maxX:
                maxX = corners[0][c][0]
            if corners[0][c][0] < minX:
                minX = corners[0][c][0]
        # print (maxY - minY)
        return maxY - minY

    def _find_biggest_detection(self, ids, corners):
        max_height = 0
        max_id = -1
        for c in range(0, len(corners)):
            h = self._get_marker_height(corners[c])
            if h > max_height:
                max_height = h
                max_id = ids[c]
        return str(max_id[0]), c, max_height

    def plot_detection(self, marker_id=-1):
        if marker_id == -1:
            marker_id = self.best_detection_id
        corners = self.detections[marker_id]['corners']
        cv2.circle(self.last_frame_RGB, tuple(corners[0][0]), 1, (0, 0, 255), -1)
        cv2.circle(self.last_frame_RGB, tuple(corners[0][1]), 1, (0, 0, 255), -1)
        cv2.circle(self.last_frame_RGB, tuple(corners[0][2]), 1, (0, 0, 255), -1)
        cv2.circle(self.last_frame_RGB, tuple(corners[0][3]), 1, (0, 0, 255), -1)

        cv2.imshow(DETECTION_WINDOW_NAME, self.last_frame_RGB)
        # cv2.waitKey(-1)

    @staticmethod
    def compute_XY_position_on_marker_detection(marker_id, tvec, rvec, annotated_map):
        marker_position_XY = annotated_map.map_landmarks_dict[marker_id][0].position
        marker_normal = annotated_map.map_landmarks_dict[marker_id][0].normal
        X, Y, Z = MarkerDetector.convert_coors_from_camera_to_marker([0., 0., 0.], rvec, tvec)
        X_line, Y_line, Z_line = MarkerDetector.convert_coors_from_camera_to_marker([0., 0., 1.], rvec, tvec)

        # marker_normal = np.array((marker_normal[1], marker_normal[0]), dtype=np.float64)
        delta_XY = X * marker_normal + np.asarray([marker_normal[1], -marker_normal[0]]) * (Z)
        line_abs = [X_line - X, Z_line - Z]

        yaw = math.atan2(line_abs[1] * marker_normal[1] + line_abs[0] * marker_normal[0],
                         line_abs[1] * marker_normal[0]
                         - line_abs[0] * marker_normal[1])

        # Alejandro convention
        #yaw = yaw - math.pi / 2.
        position_XY = marker_position_XY + delta_XY
        yaw = (yaw + 2 * math.pi) % (2 * math.pi)

        return position_XY, yaw

    @staticmethod
    def convert_coors_from_camera_to_marker(P, rvec, tvec):
        # P is the 3D point (in camera coordinates) to be converted to marker/board
        # (such as marker or charuco board) coordinate system.
        # Coordinate system is specified by rvec,tvec, returned from a marker/board detection.
        # Note that rvec,tvec are numpy arrays, not matrices
        R, _ = cv2.Rodrigues(rvec)
        R = np.matrix(R)
        P = np.matrix(P).transpose()  # create a column vector
        t = np.matrix(tvec).transpose()  # create a column vector
        # combinations I tried that failed: #P2 = R*P + t #P2 = R*(P + t) #P2 = np.linalg.inv(R)*P - t
        P2 = np.linalg.inv(R) * (P - t)
        X, Y, Z = P2[0, 0], P2[1, 0], P2[2, 0]

        return X, Y, Z