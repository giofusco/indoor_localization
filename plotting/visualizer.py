import map.annotated_map
import cv2
import numpy as np

NAVIGATION_WINDOW_NAME = 'Trajectory'
FRAME_WINDOW_NAME = 'Current Frame'


class Visualizer:

    # @staticmethod
    # def plot_yaw(image):
    #     # plot absolute VIO yaw
    #     p = self.map_data.env_map.xy2uv(self.observers[cnames.ODOMETRY].current_position)
    #     vio_y_rotation = self.observers[cnames.ODOMETRY].starting_yaw + self.current_data[dconst.CAMERA_ROTATION][
    #         1] - math.pi / 2
    #     # vio_y_rotation = self.observers[cnames.ODOMETRY].current_yaw - math.pi/2
    #     # print ">>", vio_y_rotation*180/math.pi
    #     x = int(math.cos(vio_y_rotation) * 20)
    #     y = int(math.sin(vio_y_rotation) * 20)
    #     image = cv2.arrowedLine(image, (p[0], p[1]), (p[0] - x, p[1] + y), (0, 255, 0), 2)
    #
    #     if self.observers[cnames.ODOMETRY].last_marker_yaw is not None:
    #         x = int(math.cos(self.observers[cnames.ODOMETRY].last_marker_yaw - math.pi / 2) * 20)
    #         y = int(math.sin(self.observers[cnames.ODOMETRY].last_marker_yaw - math.pi / 2) * 20)
    #         image = cv2.arrowedLine(image, (p[0], p[1]), (p[0] - x, p[1] + y), (255, 0, 0), 2)
    #     # print "** LAST MARKER YAW ", self.observers[cnames.ODOMETRY].last_marker_yaw * 180/math.pi
    #     # axis_orientation = self.observers[cnames.ODOMETRY].starting_yaw
    #     # x = int(math.cos(axis_orientation - math.pi / 2) * 20)
    #     # y = int(math.sin(axis_orientation - math.pi / 2) * 20)
    #     # image = cv2.arrowedLine(image, (p[0], p[1]), (p[0] - x, p[1] + y), (0, 0, 255), 2)
    #
    #     return image

    @staticmethod
    def show_frame(image):
        cv2.imshow("input", image)
        cv2.waitKey(1)

    @staticmethod
    def plot_measured_position_on_map(position, annotated_map, color=(0,0,255), window_name=NAVIGATION_WINDOW_NAME):
        uv = annotated_map.xy2uv(position)
        traj_image = Visualizer._plot_point(annotated_map.get_walls_image(), uv, color)
        cv2.imshow(window_name, traj_image)
        cv2.waitKey(1)

    @staticmethod
    def plot_particles(annotated_map, particles):
        if len(np.shape(annotated_map.get_walls_image())) == 2:
            draw_map = cv2.cvtColor(annotated_map.get_walls_image(), cv2.COLOR_GRAY2RGB)
        else:
            draw_map = annotated_map.get_walls_image().copy()
        for point in particles:
            cv2.circle(draw_map, tuple(annotated_map.xy2uv(point[0:2])), 1, (0, 0, 255))
        cv2.imshow("part", draw_map)
        cv2.waitKey(1)

    @staticmethod
    def _plot_point(map_image, uv, color=(0, 0, 255)):
        if len(np.shape(map_image)) == 2:
            draw_map = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
        else:
            draw_map = map_image.copy()
        draw_map = cv2.circle(draw_map, tuple(uv), 5, color, -1)
        return draw_map



    # def plot_trace(self):
    #     wall_map = self.map_data.get_walls_image()
    #     wall_map =255 - (wall_map)
    #     if wall_map.shape[2] <3:
    #         wall_map = cv2.cvtColor(wall_map, cv2.COLOR_GRAY2RGB)
    #
    #     marker_points = self.observers[cnames.ODOMETRY].marker_detection_locations
    #     # for p1, p2 in self.pairwise(self.position_trace):
    #         # wall_map = cv2.line(wall_map, tuple(p1), tuple(p2), (0, 0, 255), 3)
    #         # wall_map = cv2.circle(wall_map, tuple(p1), 5, (0, 0, 255), 1)
    #         # wall_map = cv2.circle(wall_map, tuple(p2), 5, (0, 0, 255), 1)
    #     for p in marker_points:
    #         # p = ((int(p[0]), int(p[1])))
    #         p = self.map_data.env_map.xy2uv(p)
    #         wall_map = cv2.circle(wall_map, tuple(p), 6, (0, 0, 255), 2)
    #     cv2.imshow("Trace", wall_map)
    #     cv2.imwrite(self.data_source.folder +"/trajectory.png", wall_map)
    #     cv2.waitKey(-1)
