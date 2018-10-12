import components_names as cnames
import input.data_constants as dconst
import numpy as np
import math
import cv2
import glob
import itertools
from plotting import visualizer as vis
from particlefilter.particle_filter import ParticleFilter
from particlefilter import particle_filter
from scipy.spatial.distance import cdist

from numba import jit

class NavigationSystem:

    def __init__(self, annotated_map, data_source, marker_detector, visualizer=None):
        self.data_source = data_source
        self.current_data = None
        self.observers = {}
        self.annotated_map = annotated_map
        self.map_image = annotated_map.get_walls_image()
        self.debug_images = {}
        self.last_tvec = None
        self.last_position = None
        self.last_yaw = None
        self.last_marker_yaw = None
        self.last_marker_position = None
        self.csv_file = open('data.csv', 'w')
        self.position_trace = []
        self.markers_trace = []
        self.marker_detector = marker_detector
        self.particle_filter = None
        self.visualizer = visualizer
        self.initial_position_set = False
        self.frame_counter = 0
        self.position_filename = data_source.folder + '/data_log.txt'

        # create user positions file
        fh = open(self.position_filename, "w")
        fh.write("USER POSITION\n")
        fh.close
        self.position_file_handler = open(self.position_filename, "a")


    def set_data_source(self, data_source):
        self.data_source = data_source

    def attach(self, name, observer):
        self.observers[name] = observer

    def initialize(self, num_particles=1000, init_pos_noise=0.1, step_pos_noise_maj=0.1, step_pos_noise_min=0.1, init_yaw_noise=0.1, 
                    step_yaw_noise=0.1, fudge_max = 1., check_wall_crossing=True, uniform=True):

        self.particle_filter = ParticleFilter(self.annotated_map, num_particles=num_particles, position_noise_maj=step_pos_noise_maj,
                                                position_noise_min=step_pos_noise_min, yaw_noise=step_yaw_noise,
                                                check_wall_crossing=check_wall_crossing, visualizer=self.visualizer)
        self.detect_starting_position()
        measured_pos, measured_yaw, VIO_yaw_offset = self.observers[cnames.ODOMETRY].get_initial_measurements()
       
        if not uniform:
            self.particle_filter.initialize_particles_at(measured_pos, measured_yaw, VIO_yaw_offset, init_pos_noise, init_yaw_noise, fudge_max)
        else:
            self.particle_filter.initialize_particles_uniform(init_pos_noise, init_yaw_noise, fudge_max)

    def detect_starting_position(self):
        self.observers[cnames.ODOMETRY].set_initial_position(self.data_source, self.marker_detector, verbose=True)
        self.initial_position_set = True
        print("Initialization completed.")
        # self.observers[cnames.MARKER_DETECTOR].disable()

    # execute a step
    def step(self):
        self.current_data = self.data_source.read_next(load_image=True)

        if not self.current_data == {}:
            self.frame_counter += 1
            # notify all observers that new data are available and update accordingly
            for name, observer in self.observers.items():
                observer.update(self.current_data)
            VIO_pos, measured_VIO_yaw, measured_pos_delta, yaw_delta, tracker_status = self.observers[cnames.ODOMETRY].get_measurements_and_deltas()
            observed_pos_marker, observed_yaw_marker = self.observers[cnames.MARKER_DETECTOR].get_observations(annotated_map=self.annotated_map)
            observed_sign_distance, observed_sign_roi = self.observers[cnames.EXIT_DETECTOR].get_sign_info()

            # measured_pos, measured_yaw = self.observers[cnames.ODOMETRY].get_measurements()

            self.visualizer.show_frame(self.current_data[dconst.IMAGE])

            self.particle_filter.step(measurements=[VIO_pos, measured_VIO_yaw, measured_pos_delta, yaw_delta, tracker_status],
                                      observations=[observed_pos_marker, observed_yaw_marker, observed_sign_distance,
                                                    observed_sign_roi])

            self.calculate_user_location()


            self.position_trace.append(self.particle_filter.particles[0, 0:2] + [0., 0.])
            self.position_file_handler.write(str(measured_VIO_yaw)+ ',' + str(VIO_pos[particle_filter.PF_X]) + ',' +
                                             str(VIO_pos[particle_filter.PF_Z]) + ',' +
                                             str(self.observers[cnames.ODOMETRY].starting_yaw - math.pi/2) + ',' +
                                             str(self.observers[cnames.ODOMETRY].starting_position[0]) + ',' +
                                             str(self.observers[cnames.ODOMETRY].starting_position[1]) + '\n')

            # self.position_file_handler.write(str(self.particle_filter.particles[0][particle_filter.PF_X])+ "\t"+ str(self.particle_filter.particles[0][particle_filter.PF_Z])+"\n")
            # self.position_file_handler.write(str(measured_pos_delta[particle_filter.PF_X]) + "\t" + str(
            #      measured_pos_delta[particle_filter.PF_Z]) + "\n")
            # self.position_file_handler.write(str(measured_VIO_yaw) + "\t" + str(self.particle_filter.particles[:, 2]) + "\n")

        else:
            raise RuntimeError("Out of Data to process. Ending navigation system.")

    @staticmethod
    def pairwise(self, iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    def save_start_marker_timestamp(self, data_folder):
        if (len(self.observers[cnames.ODOMETRY].time_trigger_marker_seen)>0):
            idx = np.argmax(np.diff(self.observers[cnames.ODOMETRY].time_trigger_marker_seen))
            sync_Timestamp = self.observers[cnames.ODOMETRY].time_trigger_marker_seen[idx]
            fh = open(data_folder + '/sync_ts.txt', "w")
            fh.write(str(sync_Timestamp))
            fh.close

    def finish(self):
        self.position_file_handler.close()
        self.visualizer.close_all_windows()
        self.visualizer.plot_trace(self.data_source, self.annotated_map, self.position_trace)

    def calculate_user_location(self, verbose = True):
        uv_pix = self.annotated_map.uv2pixels_vectorized(self.particle_filter.particles[:, 0:2])
        values = prepare_heat_map(uv_pix.astype(np.int32), self.particle_filter.particles[:, particle_filter.PF_SCORE],
                                  self.annotated_map.get_walls_image().shape)

        kde = cv2.GaussianBlur(values, (11, 11), 35.)
        idx_sort = np.argsort(kde, axis=1)
        idx = idx_sort[:,-1]

        idx_row = np.argsort(kde[np.arange(len(kde)), idx], axis=0)
        loc_max_0 = np.array( (idx[idx_row[-1]],idx_row[-1] ))
        max_0 = kde[idx_row[-1], idx[idx_row[-1]]]

        loc_max_1 =  np.array((idx[idx_row[-2]], idx_row[-2]))
        max_1 = kde[idx_row[-2], idx[idx_row[-2]]]

        dist = np.linalg.norm(loc_max_1-loc_max_1)
        # print(dist)
        if verbose:
            self.visualizer.visualize_heat_map(kde, loc_max_0, loc_max_1, None)
            # self.visualizer.visualize_heat_map(kde, loc_max_0, loc_max_1, self.frame_counter)

# @jit(nopython=True)
# def find_peaks(M, peaks):
#     a peak is row, col, value
    # for r in range(0, )

@jit(nopython=True)
def prepare_heat_map(px_points, scores, map_shape):

    map_image = np.zeros((map_shape[0], map_shape[1], 1), np.float32)
    for i_pns in range(0, len(px_points)):
        p = [px_points[i_pns, 1], px_points[i_pns,0]]
        w = scores[i_pns]
        map_image[p[0], p[1]] += w

    return map_image
