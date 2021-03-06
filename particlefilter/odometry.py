import input.data_constants as dconst

import numpy as np
import math
import cv2


class Odometry:

    def __init__(self, name, annotated_map, scale_compensation_factor = 0.0):
        self.name = name
        self.starting_position = None
        self.starting_yaw = None
        self.annotated_map = annotated_map
        self.scale_compensation_factor = scale_compensation_factor
        self.current_yaw = None
        self.current_position = None
        self.previous_yaw = None
        self.previous_position = None
        self.last_processed_timestamp = 0
        self.time_trigger_marker_seen = []
        self.last_VIO_pos_on_marker = []
        self.last_VIO_yaw_on_marker = None

    def set_initial_position(self, data_source, marker_detector=None, trigger_marker_id='55'):
        if marker_detector is not None:
            found = False
            position_XY, yaw = None, None
            while not found:
                current_data = data_source.read_next(load_image=True)
                if not current_data == {}:
                    if current_data[dconst.VIO_STATUS] == 'normal':
                        # print current_data[dconst.CAMERA_ROTATION][1]*180/math.pi
                        if current_data[dconst.IMAGE] is not None:
                            marker_detector.update(current_data)
                            # if we have a detection, initialize position and orientation
                            if trigger_marker_id in marker_detector.detections:
                                self.time_trigger_marker_seen.append(current_data[dconst.TIMESTAMP])

                            if marker_detector.best_detection_id is not None:
                                marker_id = marker_detector.best_detection_id
                                tvec = marker_detector.detections[marker_id]['tvec']
                                rvec = marker_detector.detections[marker_id]['rvec']

                                # if marker_id in self.map_data.env_map.map_landmarks_dict and \
                                #                 tvec[2] <= marker_detector.max_marker_distance_meter:
                                #     marker_position_XY, yaw_marker = \
                                #         marker_detector.compute_XY_position_on_marker_detection(marker_id, tvec, rvec)

                                marker_position_XY, yaw_marker = marker_detector.get_observations(self.annotated_map)
                                if marker_position_XY is not None:
                                    self.last_marker_yaw = yaw_marker
                                    self.last_marker_position = marker_position_XY
                                    # find out what is the VIO theta_0
                                    yaw = yaw_marker - current_data[dconst.CAMERA_ROTATION][1]
                                    # print ("Estimated YAW: ", yaw * 180./math.pi)
                                    found = True
                                    # print "First Marker -- id: ", marker_id
                                    self.last_processed_timestamp = current_data[dconst.TIMESTAMP]
                                    # self.dump_to_csv()
                                    self.last_VIO_pos_on_marker = [current_data[dconst.CAMERA_POSITION][0],
                                                                   current_data[dconst.CAMERA_POSITION][1]]
                                    self.last_VIO_yaw_on_marker = current_data[dconst.CAMERA_ROTATION][1]
                                    self.starting_position = self.last_marker_position
                                    self.starting_yaw = yaw
                                    #self.last_marker_position = self.last_marker_position
                                    self.current_yaw = yaw
                                    self.current_position = self.last_marker_position
                    else:
                        if current_data[dconst.IMAGE] is not None:
                            marker_detector.update(current_data)
                            if trigger_marker_id in marker_detector.detections:
                                self.time_trigger_marker_seen.append(current_data[dconst.TIMESTAMP])

                else:
                    raise ("out of data to process")


        else:
            raise ("need a marker detector to use this marker based position initialization")


    def update(self, vio_data):
        # print("***", vio_data[dconst.VIO_STATUS])
        # if vio_data[dconst.VIO_STATUS] == 'normal' or vio_data[dconst.VIO_STATUS] == 'limited':
        self._update_odometry(vio_data)
        self.last_processed_timestamp = vio_data[dconst.TIMESTAMP]

#TODO: check yaw update
    def _update_odometry(self, vio_data):

        self.previous_yaw = self.current_yaw
        self.previous_position = self.current_position

        delta_yaw = self.last_VIO_yaw_on_marker - self.last_marker_yaw
        # print "Delta YAW: ", delta_yaw * 180 / math.pi
        delta_X = vio_data[dconst.CAMERA_POSITION][0] - self.last_VIO_pos_on_marker[0]
        delta_Z = vio_data[dconst.CAMERA_POSITION][2] - self.last_VIO_pos_on_marker[1]
        delta_Z *= -1
        x = self.last_marker_position[0] + math.cos(delta_yaw) * delta_X + math.sin(delta_yaw) * delta_Z
        z = self.last_marker_position[1] - math.sin(delta_yaw) * delta_X + math.cos(delta_yaw) * delta_Z
        self.current_yaw = self.last_marker_yaw + delta_yaw
        self.current_position = np.asarray([x, z], np.float)

    def get_measurements_deltas(self):
        return self.current_position-self.previous_position, self.current_yaw-self.previous_yaw

    def get_measurements(self):
        return self.current_position, self.current_yaw