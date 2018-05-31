import numpy as np
from math import trunc
from plotting.visualizer import Visualizer
from numba import jit
import time

PF_HIT_WALL = -1
PF_NOT_VALID = -2


class ParticleFilter:

    # initialize particle filter from configuration file
    def __init__(self, config_file):
        self.particles = []
        raise NotImplementedError

    def __init__(self, annotated_map, num_particles=1000, visualizer=None):
        self.num_particles = num_particles
        self.particles = []
        self.annotated_map = annotated_map
        self.walls_image = self.annotated_map.get_walls_image()
        self.vis = visualizer

#todo: check if particle is outside of walkable area
    def initialize_particles_at(self, pos, yaw, position_noise_sigma=0.2, yaw_noise_sigma=0.1):
        #initialize N random particles all over the walkable area
        sample_x = np.random.normal(pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(pos[1], position_noise_sigma, self.num_particles)
        yaws = np.random.normal(yaw, yaw_noise_sigma, self.num_particles)
        weights = np.zeros((self.num_particles))
        self.particles = np.column_stack((sample_x, sample_y, yaws, weights))
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def step(self, measurements, observations):
        print ("Particle Filter step")
        self.move_particles_by(measurements[0], measurements[1], position_noise_sigma=0.05)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def move_particles_by(self, delta_pos, delta_yaw, position_noise_sigma=0.1):
        sample_x = np.random.normal(delta_pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(delta_pos[1], position_noise_sigma, self.num_particles)
        uv_pt1 = self.annotated_map.xy2uv_vectorized(self.particles[:, 0:2])
        uv_pt2 = self.annotated_map.xy2uv_vectorized(self.particles[:, 0:2] + np.column_stack((sample_x, sample_y)))

        t0 = time.time()
        cnt = 0
        for p, item in enumerate(self.particles):

            # cross = check_traversability(self.walls_image, int(uv_pt1[p][0]), int(uv_pt1[p][1]), int(uv_pt2[p][0]), int(uv_pt2[p][1]))
            cross = wall_hit(int(uv_pt1[p][1]), int(uv_pt1[p][0]), int(uv_pt2[p][1]),
                                         int(uv_pt2[p][0]), self.walls_image)

            if cross:
                item[3] = PF_HIT_WALL
            cnt += 1
        t1 = time.time()
        print("CheckTraversability loop:", t1 - t0)
        self.particles[:,0] += sample_x
        self.particles[:, 1] += sample_y
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)


@jit(nopython=True)
def wall_hit(r1, c1, r2, c2, m):  # improved version? need to test!
    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white (wall) pixels are hit along the way.
    Return 1 if a wall is hit, 0 otherwise."""

    dr = abs(r1 - r2)  # delta row
    dc = abs(c1 - c2)  # delta column

    span = max(dr, dc)  # if more rows than columns then loop over rows; else loop over columns
    span_float = span + 0.  # float version
    if span == 0:  # i.e., special case: (r1,c1) equals (r2,c2)
        multiplier = 0.
    else:
        multiplier = 1. / span
    for k in range(span + 1):  # k goes from 0 through span; e.g., a span of 2 implies there are 2+1=3 pixels to reach in loop
        frac = k * multiplier
        r = trunc(r1 + frac * (r2 - r1))
        c = trunc(c1 + frac * (c2 - c1))
        if m[r, c] > 0:  # hit!
            return 1  # report hit and exit function

    return 0  # if we got to here then no hits