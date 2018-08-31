import map.annotated_map
import cv2
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from math import pi, cos, sin

NAVIGATION_WINDOW_NAME = 'Trajectory'
FRAME_WINDOW_NAME = 'Current Frame'


class Visualizer:
    def __init__(self, map_image):
        self.jets = plt.get_cmap('plasma')
        if len(np.shape(map_image)) == 2:
            self.draw_map = cv2.cvtColor(map_image, cv2.COLOR_GRAY2RGB)
        else:
            self.draw_map = map_image.copy()

    @staticmethod
    def close_all_windows():
        cv2.destroyAllWindows()

    def plot_map_feature(self, annotated_map, feature_id, num_feature=None):
        draw_map = self.draw_map.copy()
        if num_feature is not None:
            pos = annotated_map.xy2uv(annotated_map.map_landmarks_dict[feature_id][num_feature].position)
            cv2.circle(draw_map, tuple(pos), 1, (128, 255, 0))
            cv2.imshow("Location of " + feature_id)
        else:
            for f in annotated_map.map_landmarks_dict[feature_id]:
                pos = annotated_map.uv2pixels(f.position)
                cv2.circle(draw_map, tuple((pos[0], pos[1])), 1, (128, 255, 0))
                cv2.putText(draw_map,str(f.position),tuple( (pos[0], pos[1])),cv2.FONT_HERSHEY_COMPLEX_SMALL,.5, (0,255,0))
            cv2.imshow("Locations of " + feature_id, draw_map)
            cv2.waitKey(-1)

    def plot_particle_displacement(self, annotated_map, particles, destinations_px):
        draw_map = self.draw_map.copy()
        pos = (annotated_map.xy2uv_vectorized(particles[:, 0:2]))
        # dest = annotated_map.xy2uv_vectorized(destinations_uv)
        # dest = (annotated_map.xy2uv_vectorized(destinations[:, 0:2]))
        # deltas = dest - pos

        for p in range(len(particles)):
            cv2.circle(draw_map, tuple(pos[p]), 1,
                       (0, 255, 0))
            # yaw = particles[p][2]
            # x = int(cos(yaw) * (delta_uv[0]))
            # y = int(sin(yaw) * (delta_uv[1]))
            cv2.circle(draw_map, (destinations_px[p][0], destinations_px[p][1]), 2, (0, 0, 255))
            # cv2.circle(draw_map, tuple(annotated_map.xy2uv(destinations[p][0:2])), 5, (0, 0, 255))
            # pos = tuple(annotated_map.xy2uv(destinations[p][0:2]))
            draw_map = cv2.arrowedLine(draw_map, (pos[p][0], pos[p][1]), (destinations_px[p][0], destinations_px[p][1]), (0, 255, 0), 2)

        cv2.imshow("DISPLACEMENT", draw_map)
        # cv2.waitKey(10)

    # @staticmethod
    def show_frame(self, image):
        cv2.imshow("input", image)
        # cv2.waitKey(1)

    # @staticmethod
    def plot_measured_position_on_map(self, position, annotated_map, color=(0,0,255), window_name=NAVIGATION_WINDOW_NAME):
        uv = annotated_map.xy2uv(position)
        traj_image = self._plot_point(uv, color)
        cv2.imshow(window_name, traj_image)
        # cv2.waitKey(1)

    def plot_particles(self, annotated_map, particles):
        draw_map = self.draw_map.copy()
        valid_particles = particles[particles[:,3]>=0]
        score_colors = self.jets(valid_particles[:,3])*255
        pts = annotated_map.uv2pixels_vectorized(valid_particles[:,0:2])
        yaw_uv = valid_particles[:,0:2] + np.asarray( (np.cos(valid_particles[:,2]), np.sin(valid_particles[:,2]) ), dtype=np.float64 ).transpose()
        yaw_pts = annotated_map.uv2pixels_vectorized(yaw_uv)

        for p in range(len(valid_particles)):

            cv2.circle(draw_map, (pts[p,0], pts[p,1]), int(5*(valid_particles[p,3])),
                       (score_colors[p,2], score_colors[p,1], score_colors[p,0]))

            # draw_map = cv2.arrowedLine(draw_map, (pts[p,0], pts[p,1]), (yaw_pts[p,0], yaw_pts[p,1]), (score_colors[p,2], score_colors[p,1], score_colors[p,0]), 2)

            # if point[3] > 0.5:
            # cv2.circle(draw_map, tuple(annotated_map.xy2uv(point[0:2])), 1, (0, 255, 0))
            # else:
            #     cv2.circle(draw_map, tuple(annotated_map.xy2uv(point[0:2])), 1, (0, 0, 255))
        cv2.imshow("Particles", draw_map)
        # cv2.waitKey(-1)

    def draw_points(self, annotated_map, points, size=3, color=(0, 0, 255), title="Points"):
        draw_map = self.draw_map.copy()

        for p in range(len(points)):
            cv2.circle(draw_map, tuple(points[p][0:2]), int(size), color)
        cv2.imshow(title, draw_map)

    def plot_points(self, annotated_map, points, size=3, color=(0, 0, 255), title="Points"):
        draw_map = self.draw_map.copy()

        for p in range(len(points)):
            cv2.circle(draw_map, tuple(annotated_map.xy2uv(points[p][0:2])), int(size), color)
        cv2.imshow(title, draw_map)
        # cv2.waitKey(1)

    # @staticmethod
    def _plot_point(self, uv, color=(0, 0, 255)):
        draw_map = self.draw_map.copy()
        draw_map = cv2.circle(draw_map, tuple(uv), 5, color, -1)
        return draw_map

    def get_color_from_float(self, value):
        jet = self.jets(value)
        r = jet[0] * 255
        g = jet[1] * 255
        b = jet[2] * 255
        color = (b, g, r)
        return color


    def visualize_heat_map(self, kde_map, loc=None, loc2=None, avg = False):

        # VIO_Theta, VIO_X, VIO_Z, Marker_Theta, MARKER_X, MARKER_Z
        min_val, max_val, a, b = cv2.minMaxLoc(kde_map)
        adjMap = np.array((kde_map - min_val) * (255 / (max_val - min_val)), dtype=np.uint8)
        adjMap = cv2.applyColorMap(adjMap, cv2.COLORMAP_JET)

        if avg and loc is not None and loc2 is not None:
            avg_loc = (loc+loc2 /2).astype(np.int)
            adjMap = cv2.circle(adjMap, tuple(avg_loc), 3, (128, 128, 128), 5)

        else:
            if loc is not None:
                adjMap = cv2.circle(adjMap,tuple(loc),3, (128,128,0), 7)

            if loc2 is not None:
                adjMap = cv2.circle(adjMap,tuple(loc2),3, (0,128,128), 4)

        cv2.imshow("KDE", self.draw_map + adjMap)

    def plot_trace(self, data_source, annotated_map, trace):
        wall_map = self.draw_map.copy()
        wall_map =255 - (wall_map)
        if wall_map.shape[2] <3:
            wall_map = cv2.cvtColor(wall_map, cv2.COLOR_GRAY2RGB)


        for p in range(0, len(trace)):
            cv2.circle(wall_map, tuple(annotated_map.uv2pixels(trace[p][0:2])), int(5), (0,0,255))

        #marker_points = self.observers[cnames.ODOMETRY].marker_detection_locations
        # for p1, p2 in self.pairwise(self.position_trace):
        #     print (p1)
            # wall_map = cv2.line(wall_map, tuple(p1), tuple(p2), (0, 0, 255), 3)
            # wall_map = cv2.circle(wall_map, tuple(p1), 5, (0, 0, 255), 1)
            # wall_map = cv2.circle(wall_map, tuple(p2), 5, (0, 0, 255), 1)
        #for p in marker_points:
            # p = ((int(p[0]), int(p[1])))
          #  p = self.map_data.env_map.xy2uv(p)
           # wall_map = cv2.circle(wall_map, tuple(p), 6, (0, 0, 255), 2)
        cv2.imshow("Trace", wall_map)
        cv2.imwrite(data_source.folder +"/trajectory.png", wall_map)
