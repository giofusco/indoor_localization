import numpy as np
from math import trunc, pi
from plotting.visualizer import Visualizer
from numba import jit
import time
from random import random
import timeit
from scipy.spatial.distance import cdist

PF_HIT_WALL = -1
PF_NOT_VALID = -2

PF_X = 0
PF_Z = 1
PF_YAW = 2
PF_SCORE = 3

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
        self.walkable_mask = self.annotated_map.get_walkable_mask()
        self.vis = visualizer
        self.yaw_offset = 0

#todo: check if particle is outside of walkable area
    def initialize_particles_at(self, pos, yaw, position_noise_sigma=0.5, yaw_noise_sigma=0.1):
        #initialize N random particles all over the walkable area
        sample_x = np.random.normal(pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(pos[1], position_noise_sigma, self.num_particles)
        yaws = np.random.normal(yaw, yaw_noise_sigma, self.num_particles)
        weights = np.zeros((self.num_particles))
        self.particles = np.column_stack((sample_x, sample_y, yaws, weights))
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def initialize_particles_uniform(self, position_noise_sigma=0.5, yaw_noise_sigma=0.1):
        # initialize N random particles all over the walkable area
        sample_x = np.random.uniform(0., self.annotated_map.mapsize_xy[1], self.num_particles)
        sample_y = np.random.uniform(0., self.annotated_map.mapsize_xy[0], self.num_particles)
        yaws = np.random.normal(0, yaw_noise_sigma, self.num_particles)
        weights = np.ones(self.num_particles)
        self.particles = np.column_stack((sample_x, sample_y, yaws, weights))
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def initialize_particles_uniform_with_yaw(self, yaw_offset, position_noise_sigma=0.5, yaw_noise_sigma=0.005):
        # initialize N random particles all over the walkable area

        n_valid_particles = 0
        num_samples = self.num_particles

        while n_valid_particles < self.num_particles:
            sample_x = np.random.uniform(0., self.annotated_map.mapsize_xy[1], num_samples)
            sample_z = np.random.uniform(0., self.annotated_map.mapsize_xy[0], num_samples)
            sample_x, sample_z, num_valid = self.remove_non_walkable_locations(sample_x, sample_z)
            n_valid_particles += num_valid
            yaws = yaw_offset + np.random.normal(0, yaw_noise_sigma, num_valid)
            weights = np.ones(num_valid)
            num_samples = self.num_particles - n_valid_particles
            if len(self.particles) == 0:
                self.particles = np.column_stack((sample_x, sample_z, yaws, weights))
            else:
                self.particles = np.concatenate((self.particles, np.column_stack((sample_x, sample_z, yaws, weights))),
                                                axis=0)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)


    def step(self, measurements_deltas, observations):
        # print ("Particle Filter step")
        self.move_particles_by(measurements_deltas[0], measurements_deltas[1],
                               position_noise_sigma=0.01, yaw_noise_sigma=0.01)
        # self.score_particles(observations)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)
        # t0 = time.time()
        self.resample_particles()
        # t1 = time.time()
        # print(t1-t0)

    def remove_non_walkable_locations(self, x_vector, z_vector):
        tmp_pos = np.column_stack((x_vector, z_vector))
        tmp_pt = self.annotated_map.xy2uv_vectorized(tmp_pos)

        sample_x = x_vector[self.walkable_mask[np.array(tmp_pt[:, 1]), np.array(tmp_pt[:, 0])] > 0]
        sample_z = z_vector[self.walkable_mask[np.array(tmp_pt[:, 1]), np.array(tmp_pt[:, 0])] > 0]

        num_valid = len(sample_x)

        return sample_x, sample_z, num_valid

    def move_particles_by(self, delta_pos, delta_yaw, position_noise_sigma=0.1, yaw_noise_sigma=0.005,
                          check_wall_crossing=True):

        start_pt = (self.annotated_map.xy2uv_vectorized(self.particles[:, PF_X:PF_YAW]))

        pos_noise = self.__get_direction_noise(delta_pos, len(self.particles), sigma_pos=position_noise_sigma)
        x = self.particles[:, PF_X] + delta_pos[0] * np.cos(self.particles[:,PF_YAW])  - delta_pos[1] * np.sin(self.particles[:,PF_YAW]) #+ pos_noise[0]
        z = self.particles[:, PF_Z] - (delta_pos[0] * np.sin(self.particles[:, PF_YAW])  - delta_pos[1] * np.cos(self.particles[:, PF_YAW])) #- pos_noise[1]

        # delta = np.linalg.norm(delta_pos, ord=2) + np.abs(np.random.normal(0, position_noise_sigma, len(self.particles)))
        # x = self.particles[:, PF_X] - np.sin(self.particles[:, PF_YAW]) * delta
        # z = self.particles[:, PF_Z] + np.cos(self.particles[:, PF_YAW]) * delta
        dest_pt = self.annotated_map.xy2uv_vectorized(np.column_stack((x, z)))

        # self.vis.draw_points(self.annotated_map, start_pt, size=3, color=(0, 0, 255), title="Points - start")
        # self.vis.draw_points(self.annotated_map, dest_pt, size = 3, color = (0, 0, 255), title = "Points")

        new_yaws = self.particles[:, 2] + np.random.normal(0, yaw_noise_sigma, len(self.particles))

        # self.vis.plot_particle_displacement(annotated_map=self.annotated_map, particles=self.particles,
        #                                        destinations_uv=dest_pt)

        if check_wall_crossing:
            self.particles = set_wall_hit_score(self.particles, start_pt, dest_pt, self.walls_image)
            x = x[self.particles[:, PF_SCORE] >= 0]
            z = z[self.particles[:, PF_SCORE] >= 0]
            new_yaws = new_yaws[self.particles[:, PF_SCORE] >= 0]
            self.particles = self.particles[self.particles[:, PF_SCORE] >= 0]

        self.particles[:, PF_X] = x
        self.particles[:, PF_Z] = z
        self.particles[:, PF_YAW] = new_yaws

    def score_particles(self, observations):
        if observations[0] is not None:
            d = cdist(self.particles[:, 0:2], [observations[0]], metric='euclidean')
            self.particles[self.particles[:,PF_SCORE]>=0., PF_SCORE] = (1/(1+d[self.particles[:, PF_SCORE] >= 0.])).transpose()

    def resample_particles(self):
        # if len(self.particles) > 0:
        tot_score = np.sum(self.particles[:, PF_SCORE])
        w = self.particles[:, PF_SCORE]

        if tot_score > 0:
            w /= tot_score
        idx = np.random.choice(len(w), self.num_particles, [w])
        new_particles = self.particles[idx]
        self.particles = new_particles
        self.particles[:, PF_SCORE] = 1.
        # self.particles[:, PF_YAW] += np.random.normal(0, yaw_noise_sigma, self.num_particles)
        self.particles[:, PF_X] += np.random.normal(0, 0.1, self.num_particles)
        self.particles[:, PF_Z] += np.random.normal(0, 0.1, self.num_particles)
        print(len(self.particles))
        # else:
        #     raise RuntimeError("No particles left")

    @staticmethod
    def __get_direction_noise(delta_pos, num_samples, sigma_pos=0.1):
        tot_delta = np.sqrt(np.sum(delta_pos*delta_pos))
        x_ratio = 2*(abs(delta_pos[0]) / tot_delta)
        y_ratio = 2*(abs(delta_pos[1]) / tot_delta)

        # if abs(delta_pos[0]) < 0.025:
        #     x_ratio = 0
        # if abs(delta_pos[1]) < 0.025:
        #     y_ratio = 0

        noise = np.random.normal(0, sigma_pos, num_samples)
        return abs(noise * x_ratio), abs(noise * y_ratio)
        #return np.sign(delta_pos[0])*abs(noise*x_ratio), np.sign(delta_pos[1])*abs(noise*y_ratio)

@jit(nopython=True)
def set_wall_hit_score(particles, uv_pt1, uv_pt2, m):

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