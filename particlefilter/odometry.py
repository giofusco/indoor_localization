import input.data_constants as dconst
import numpy as np
import math


class Odometry:

    def __init__(self, name, annotated_map, scale_compensation_factor=0.0):
        self.name = name

        self.annotated_map = annotated_map
        self.scale_compensation_factor = scale_compensation_factor
        # self.current_yaw = None

        # marker inferred position and yaw of the camera
        self.current_position = None
        self.starting_position = None
        self.starting_yaw = None

        self.last_processed_timestamp = 0
        self.time_trigger_marker_seen = []

        self.measurements_deltas = [0., 0., 0., 0.]

        # raw VIO data
        self.current_VIO_position = None
        self.previous_VIO_position = None
        self.current_VIO_yaw = None
        self.previous_VIO_yaw = None
        self.VIO_yaw_offset = None
        # deltas for particles update
        self.delta_VIO_yaw = None
        self.delta_VIO_position = None
        self.current_abs_yaw = None
        self.initial_VIO_position = None

        # difference between known initial absolute yaw (given by marker for now)
        self.yaw_offset = 0

    def set_initial_position(self, data_source, marker_detector=None, trigger_marker_id='55', verbose=False):
        if marker_detector is not None:
            found = False
            position_XY, yaw = None, None
            detection_interrupted = False
            observed_yaws = []
            # measured_yaws = []
            cnt = 0
            while not found or not detection_interrupted:
                current_data = data_source.read_next(load_image=True)
                cnt += 1
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
                                #tvec = marker_detector.detections[marker_id]['tvec']
                                #rvec = marker_detector.detections[marker_id]['rvec']

                                marker_position_XY, yaw_marker = marker_detector.get_observations(self.annotated_map)

                                if marker_position_XY is not None:
                                    if verbose:
                                        marker_detector.plot_detection()
                                        print("Marker Yaw = ", yaw_marker*180/math.pi)
                                        print("VIO Yaw = ", current_data[dconst.CAMERA_ROTATION][1]*180/math.pi)
                                    # find out what is the VIO theta_0
                                    # yaw = yaw_marker - current_data[dconst.CAMERA_ROTATION][1]
                                    found = True
                                    observed_yaws.append(current_data[dconst.CAMERA_ROTATION][1] - yaw_marker)
                                    # measured_yaws.append(current_data[dconst.CAMERA_ROTATION][1])
                                    self.last_processed_timestamp = current_data[dconst.TIMESTAMP]
                                    self.starting_position = marker_position_XY
                                    self.starting_yaw = yaw_marker
                                    self.current_position = marker_position_XY
                                    self.current_VIO_position = np.array([current_data[dconst.CAMERA_POSITION][0],
                                                                 current_data[dconst.CAMERA_POSITION][2]])
                                    self.current_VIO_yaw = current_data[dconst.CAMERA_ROTATION][1]
                                    self.VIO_yaw_offset = current_data[dconst.CAMERA_ROTATION][1] - yaw_marker
                                    self.current_abs_yaw = yaw_marker

                            elif found is True:
                                detection_interrupted = True

                                self.initial_VIO_position = self.current_VIO_position
                                print("VIO Yaw: ", self.current_VIO_yaw*180./math.pi)
                                print("Initial YAW offset (theta* - phi*): ", self.VIO_yaw_offset * 180 / math.pi)
                                print ("MARKER YAW: ", self.starting_yaw * 180/math.pi)
                    else:
                        if current_data[dconst.IMAGE] is not None:
                            marker_detector.update(current_data)
                            if trigger_marker_id in marker_detector.detections:
                                self.time_trigger_marker_seen.append(current_data[dconst.TIMESTAMP])
                else:
                    raise ("out of data to process")
            print("Used " + str(cnt) + " frames to initialize.")
        else:
            raise ("need a marker detector to use this marker based position initialization")

    # update the odometry consuming the next VIO data
    def update(self, vio_data):
        #if vio_data[dconst.VIO_STATUS] == 'normal' or vio_data[dconst.VIO_STATUS] == 'limited':
        self._update_odometry(vio_data)
        self.last_processed_timestamp = vio_data[dconst.TIMESTAMP]

    # private function that updates the odometry variables
    def _update_odometry(self, vio_data):

        self.previous_VIO_position = self.current_VIO_position
        self.previous_VIO_yaw = self.current_VIO_yaw
        self.current_VIO_yaw = vio_data[dconst.CAMERA_ROTATION][1]
        self.current_VIO_position = np.array([vio_data[dconst.CAMERA_POSITION][0], vio_data[dconst.CAMERA_POSITION][2]])
        self.delta_VIO_yaw = self.current_VIO_yaw - self.previous_VIO_yaw
        self.delta_VIO_position = self.previous_VIO_position - self.current_VIO_position
        self.current_abs_yaw += self.delta_VIO_yaw
        print("VIO Pos: ",  self.current_VIO_position)

    #returns raw VIO measurements
    def get_measurements(self):
        return self.current_VIO_position, self.current_VIO_yaw

    # returns change in VIO position and yaw
    def get_measurements_and_deltas(self):
        return self.current_VIO_position, self.current_VIO_yaw, self.delta_VIO_position, self.delta_VIO_yaw

    # returns the initial position and yaw measured when looking for a marker in the beginning
    def get_initial_measurements(self):
        return self.starting_position, self.starting_yaw, self.VIO_yaw_offset
