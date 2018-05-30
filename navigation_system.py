import components_names as cnames
import input.data_constants as dconst
import numpy as np
import math
import cv2
import glob
import itertools
from plotting.visualizer import Visualizer
from particlefilter.particle_filter import ParticleFilter




class NavigationSystem:

    def __init__(self, annotated_map, data_source, marker_detector):
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

    def set_data_source(self, data_source):
        self.data_source = data_source

    def attach(self, name, observer):
        self.observers[name] = observer

    def initialize(self, num_particles=1000):
        self.particle_filter = ParticleFilter(self.annotated_map, num_particles=num_particles)
        self.detect_starting_position()
        measured_pos, measured_yaw = self.observers[cnames.ODOMETRY].get_measurements()
        # initialize particles
        self.particle_filter.initialize_particles_at(measured_pos, measured_yaw, 0.5, 0.1)

    def detect_starting_position(self):
        self.observers[cnames.ODOMETRY].set_initial_position(self.data_source, self.marker_detector)

    # execute a step
    def step(self):
        self.current_data = self.data_source.read_next(load_image=True)

        if not self.current_data == {}:

            # notify all observers that new data are available and update accordingly
            for name, observer in self.observers.items():
                observer.update(self.current_data)

            # todo: fix yaw over time
            measured_pos_delta, measured_yaw_delta = self.observers[cnames.ODOMETRY].get_measurements_deltas()
            observed_pos, observed_yaw = self.observers[cnames.MARKER_DETECTOR].get_observations(annotated_map=self.annotated_map)
            measured_pos, measured_yaw = self.observers[cnames.ODOMETRY].get_measurements()
            Visualizer.show_frame(self.current_data[dconst.IMAGE])

            Visualizer.plot_measured_position_on_map(measured_pos, self.annotated_map)
            if observed_pos is not None:
                Visualizer.plot_measured_position_on_map(observed_pos, self.annotated_map, color=(255,0,0))


            self.particle_filter.step(measurements=[measured_pos_delta, measured_yaw_delta],
                                      observations=[observed_pos, observed_yaw])
            uv = self.annotated_map.xy2uv(self.observers[cnames.ODOMETRY].current_position)
            self.position_trace.append(uv)

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

    # def finish(self):
        # self.position_file_handler.close()
        # self.plot_trace()


