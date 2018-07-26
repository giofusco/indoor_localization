from input.data_parser import DataParser
from map.annotated_map import AnnotatedMap

# import data_constants as dc
import components_names
from particlefilter.odometry import Odometry
from navigation_system import NavigationSystem
from detectors.marker_detector import MarkerDetector
from detectors.sign_detector import SignDetector
from plotting.visualizer import Visualizer
# from EnvironmentMap.MapAnnotation import MapAnnotation
# from sign_detector import SignDetector

import cv2
# import numpy as np
# import cProfile

AUTO_DETECT_STARTING_POINT = 1
STEP_PAUSE = 1
UNIFORM = 0

# 99S undershooting
data_folder = './data/82M'
map_featsfile = './res/mapFeatures.yml'
map_image = './res/Walls.png'
walkable_image = './res/Walkable.png'


# map_image = './res/lighthouse_map.png'
# map_file = None
# map_featsfile = './res/lighthouseFeatures.yml'
# scale = 55/1.826  # meter/pixel

def main():

    # reads data from VIO files
    data_parser = DataParser(data_folder)

    annotated_map = AnnotatedMap(map_image, walkable_image, map_featsfile)
    visualizer = Visualizer(annotated_map.get_walls_image())
    # visualizer.plot_map_feature(annotated_map, 'exit_sign', None)

    sign_detector = SignDetector(components_names.EXIT_DETECTOR)
    marker_detector = MarkerDetector(components_names.MARKER_DETECTOR, min_consecutive_frames=2)
    nav_system = NavigationSystem(data_source=data_parser, annotated_map=annotated_map,
                                  marker_detector=marker_detector, visualizer=visualizer )

    odometry = Odometry(components_names.ODOMETRY, annotated_map)
    nav_system.attach(components_names.ODOMETRY, odometry)
    nav_system.attach(components_names.MARKER_DETECTOR, marker_detector)
    nav_system.attach(components_names.EXIT_DETECTOR, sign_detector)

    # system initialization - scan for a marker to find the initial user location
    # if AUTO_DETECT_STARTING_POINT:
    # nav_system.detect_starting_position()
    nav_system.initialize(num_particles=10000, uniform=UNIFORM)
    while True:
        try:
            nav_system.step()
            key = cv2.waitKey(STEP_PAUSE)
            if key == 27:
                break
        except RuntimeError:
            print ("\nDone.")
            # nav_system.save_start_marker_timestamp(data_folder=data_folder)
            # nav_system.finish()
            break

if __name__ == "__main__":
    main()
