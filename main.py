from input.data_parser import DataParser
from map.annotated_map import AnnotatedMap

# import data_constants as dc
import components_names
from particlefilter.odometry import Odometry
from navigation_system import NavigationSystem
from detectors.marker_detector import MarkerDetector
from plotting.visualizer import Visualizer
# from EnvironmentMap.MapAnnotation import MapAnnotation
# from sign_detector import SignDetector

import cv2
# import numpy as np
# import cProfile

AUTO_DETECT_STARTING_POINT = 1
STEP_PAUSE = 1

data_folder = './data/98S'
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

    marker_detector = MarkerDetector(components_names.MARKER_DETECTOR, min_consecutive_frames=1)
    nav_system = NavigationSystem(data_source=data_parser, annotated_map=annotated_map,
                                  marker_detector=marker_detector, visualizer=visualizer )

    odometry = Odometry(components_names.ODOMETRY, annotated_map)
    nav_system.attach(components_names.ODOMETRY, odometry)
    nav_system.attach(components_names.MARKER_DETECTOR, marker_detector)

    # system initialization - scan for a marker to find the initial user location
    if AUTO_DETECT_STARTING_POINT:
        nav_system.detect_starting_position()
    nav_system.initialize(10000)
    while True:
        try:
            nav_system.step()
            cv2.waitKey(STEP_PAUSE)
        except RuntimeError:
            print ("\nDone.")
            nav_system.save_start_marker_timestamp(data_folder=data_folder)
            # nav_system.finish()
            break

if __name__ == "__main__":
    main()
