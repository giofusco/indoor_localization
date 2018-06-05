import numpy as np
from math import trunc
from plotting.visualizer import Visualizer
from numba import jit
import time
from random import random
import timeit
from scipy.spatial.distance import cdist

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
    def initialize_particles_at(self, pos, yaw, position_noise_sigma=0.5, yaw_noise_sigma=0.1):
        #initialize N random particles all over the walkable area
        sample_x = np.random.normal(pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(pos[1], position_noise_sigma, self.num_particles)
        yaws = np.random.normal(yaw, yaw_noise_sigma, self.num_particles)
        weights = np.zeros((self.num_particles))
        self.particles = np.column_stack((sample_x, sample_y, yaws, weights))
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def step(self, measurements, observations):
        # print ("Particle Filter step")
        self.move_particles_by(measurements[0], measurements[1], position_noise_sigma=0.05)
        self.score_particles(observations)
        # t0 = time.time()
        self.resample_particles()
        # t1 = time.time()
        # print(t1-t0)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)


    def move_particles_by(self, delta_pos, delta_yaw, position_noise_sigma=0.1):
        sample_x, sample_y = self.__get_direction_noise(delta_pos, len(self.particles), sigma_x=0.1, sigma_y=0.1)
        uv_pt1 = self.annotated_map.xy2uv_vectorized(self.particles[:, 0:2])
        uv_pt2 = self.annotated_map.xy2uv_vectorized(self.particles[:, 0:2] + np.column_stack((sample_x, sample_y)))


        self.particles = wall_hit(self.particles, uv_pt1, uv_pt2, self.walls_image)
        sample_x = sample_x[self.particles[:, 3] >= 0]
        sample_y = sample_y[self.particles[:, 3] >= 0]
        self.particles = self.particles[self.particles[:, 3] >= 0]
        self.particles[:, 0] += sample_x
        self.particles[:, 1] += sample_y


    def score_particles(self, observations):

        if observations[0] is not None:
            d = cdist(self.particles[:, 0:2], [observations[0]], metric='euclidean')
            self.particles[self.particles[:,3]>=0,3] = (1/(1+d[self.particles[:,3]>=0])).transpose()

    def resample_particles(self):
        new_particles = np.zeros(np.shape(self.particles))
        # print(len(self.particles))
        w = self.particles[:,3] / np.sum(self.particles[:,3])
        idx = np.random.choice(len(w), self.num_particles, [w])
        new_particles = self.particles[idx]
        self.particles = new_particles


    @staticmethod
    def __get_direction_noise(delta_pos, num_samples, sigma_x=0.1, sigma_y=0.1):
        tot_delta = np.sqrt(np.sum(delta_pos*delta_pos))
        x_ratio = (abs(delta_pos[0]) / tot_delta) * sigma_x
        y_ratio = (abs(delta_pos[1]) / tot_delta) * sigma_y

        if abs(delta_pos[0]) < 0.01:
            x_ratio = 0
        if abs(delta_pos[1]) < 0.01:
            y_ratio = 0

        sample_x = np.random.normal(delta_pos[0], x_ratio, num_samples)
        sample_y = np.random.normal(delta_pos[1], y_ratio, num_samples)

        return np.sign(delta_pos[0])*abs(sample_x), np.sign(delta_pos[1])*abs(sample_y)

@jit(nopython=True)
def wall_hit(particles, uv_pt1, uv_pt2, m):  # improved version? need to test!

    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white (wall) pixels are hit along the way.
    Return 1 if a wall is hit, 0 otherwise."""
    for p in range(0, len(particles)):
        r1 = int(uv_pt1[p][1])
        c1 = int(uv_pt1[p][0])
        r2 = int(uv_pt2[p][1])
        c2 = int(uv_pt2[p][0])

        dr = abs(r1 - r2)  # delta row
        dc = abs(c1 - c2)  # delta column

        span = max(dr, dc)  # if more rows than columns then loop over rows; else loop over columns
        # span_float = span + 0.  # float version
        if span == 0:  # i.e., special case: (r1,c1) equals (r2,c2)
            multiplier = 0.
        else:
            multiplier = 1. / span

        for k in range(span + 1):  # k goes from 0 through span; e.g., a span of 2 implies there are 2+1=3 pixels to reach in loop
            frac = k * multiplier
            r = trunc(r1 + frac * (r2 - r1))
            c = trunc(c1 + frac * (c2 - c1))
            if m[r, c] > 0:  # hit!
                particles[p][3] = PF_HIT_WALL  # report hit and exit function
                break
    return particles