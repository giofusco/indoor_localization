import numpy as np
from math import trunc, pi, cos, sin, acos, atan2
from numba import jit
from scipy.spatial.distance import cdist
import camera_info

PF_HIT_WALL = -1
PF_NOT_VALID = -2
PF_X = 0
PF_Z = 1
PF_U = 0
PF_V = 1
PF_YAW = 2
PF_SCORE = 3
PF_YAW_OFFSET = 4
PF_FUDGE = 5

PF_VIO_POS = 0
PF_VIO_YAW = 1
PF_DELTA_POS = 2
PF_DELTA_YAW = 3
PF_TRACKER_STATUS = 4


class ParticleFilter:

    # initialize particle filter from configuration file
    def __init__(self, config_file):
        self.num_particles = None
        self.particles = []
        self.annotated_map = None
        self.walls_image = self.annotated_map.get_walls_image()
        self.walkable_mask = self.annotated_map.get_walkable_mask()
        self.vis = None
        self.yaw_offset = None
        self.tot_motion = 0.
        self.position_noise_maj = 0.
        self.position_noise_min = 0.
        self.yaw_noise = 0.
        self.check_wall_crossing = None
        raise NotImplementedError

    def __init__(self, annotated_map, num_particles=1000, position_noise_maj=1.2, position_noise_min=0.2, yaw_noise=0.01, check_wall_crossing=True, visualizer=None):
        self.num_particles = num_particles
        self.particles = []
        self.annotated_map = annotated_map
        self.walls_image = self.annotated_map.get_walls_image()
        self.walkable_mask = self.annotated_map.get_walkable_mask()
        self.vis = visualizer
        self.yaw_offset = None
        self.tot_motion = 0.
        self.position_noise_maj = position_noise_maj
        self.position_noise_min = position_noise_min
        self.yaw_noise = yaw_noise
        self.check_wall_crossing = check_wall_crossing

    def initialize_particles_at(self, pos, global_yaw, yaw_offset, position_noise_sigma, yaw_noise_sigma, fudge_max=1.):
        #initialize N random particles all over the walkable area

        self.yaw_offset = yaw_offset + pi / 2


        n_valid_particles = 0
        num_samples = self.num_particles

        while n_valid_particles < self.num_particles:
            sample_x = np.random.normal(pos[0], position_noise_sigma, num_samples)
            sample_z = np.random.normal(pos[1], position_noise_sigma, num_samples)

            sample_x, sample_z, num_valid = self.remove_non_walkable_locations(sample_x, sample_z)
            n_valid_particles += num_valid
            # yaws = yaw_offset + np.random.normal(0, yaw_noise_sigma, num_valid)
            # yaws = yaw_offset + np.random.normal(0, 0, num_valid)

            weights = np.ones(num_valid)

            num_samples = self.num_particles - n_valid_particles
            yaws = global_yaw * np.ones((num_valid, 1), dtype=np.float64)
            yaws = yaws.squeeze()

            if len(self.particles) == 0:
                self.particles = np.column_stack((sample_x, sample_z, yaws, weights,  self.yaw_offset +
                                          np.random.normal(0, yaw_noise_sigma, num_valid),
                                          np.random.uniform(1, fudge_max, num_valid)))
            else:
                self.particles = np.concatenate(
                    (self.particles, np.column_stack((sample_x, sample_z, yaws, weights,  self.yaw_offset +
                                          np.random.normal(0, yaw_noise_sigma, num_valid),
                                          np.random.uniform(1, fudge_max, num_valid) ) ) ),
                    axis=0)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)


    def initialize_particles_uniform_with_yaw(self, global_yaw, yaw_offset, position_noise_sigma=0.5, yaw_noise_sigma=0.1,
                                              fudge_max = 1.):
        # initialize N random particles all over the walkable area

        self.yaw_offset = yaw_offset + pi / 2

        n_valid_particles = 0
        num_samples = self.num_particles

        while n_valid_particles < self.num_particles:
            sample_x = np.random.uniform(0., self.annotated_map.mapsize_xy[1], num_samples)
            sample_z = np.random.uniform(0., self.annotated_map.mapsize_xy[0], num_samples)

            num_valid = self.num_particles

            sample_x, sample_z, num_valid = self.remove_non_walkable_locations(sample_x, sample_z)
            n_valid_particles += num_valid

            weights = np.ones(num_valid)

            num_samples = self.num_particles - n_valid_particles
            yaws = global_yaw * np.ones((num_valid, 1), dtype=np.float64)
            yaws = yaws.squeeze()

            if len(self.particles) == 0:
                self.particles = np.column_stack((sample_x, sample_z, yaws, weights, self.yaw_offset +
                                                  np.random.normal(0, yaw_noise_sigma, num_valid),
                                                  np.random.uniform(1, fudge_max, num_valid)))
            else:
                self.particles = np.concatenate(
                    (self.particles, np.column_stack((sample_x, sample_z, yaws, weights, self.yaw_offset +
                                                      np.random.normal(0, yaw_noise_sigma, num_valid),
                                                      np.random.uniform(1, fudge_max, num_valid)))),
                    axis=0)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)


    def initialize_particles_uniform(self, position_noise_sigma=0.5, yaw_noise_sigma=0.1,
                                              fudge_max = 1.):
        # initialize N random particles all over the walkable area

        self.yaw_offset = 0

        n_valid_particles = 0
        num_samples = self.num_particles

        while n_valid_particles < self.num_particles:
            sample_x = np.random.uniform(0., self.annotated_map.mapsize_xy[1], num_samples)
            sample_z = np.random.uniform(0., self.annotated_map.mapsize_xy[0], num_samples)

            sample_x, sample_z, num_valid = self.remove_non_walkable_locations(sample_x, sample_z)
            n_valid_particles += num_valid

            weights = np.ones(num_valid)

            num_samples = self.num_particles - n_valid_particles
            yaws = np.random.uniform(0, pi*2, num_valid)
            yaw_offset = np.random.uniform(0, pi * 2, num_valid)
            yaws = yaws.squeeze()

            if len(self.particles) == 0:
                self.particles = np.column_stack((sample_x, sample_z, yaws, weights, yaw_offset,
                                                  np.random.uniform(1, fudge_max, num_valid)))
            else:
                self.particles = np.concatenate(
                    (self.particles, np.column_stack((sample_x, sample_z, yaws, weights, yaw_offset,
                                                      np.random.uniform(1, fudge_max, num_valid)))),
                    axis=0)
        self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def step(self, measurements, observations):
        # print(" >>> Number of particles: ", len(self.particles))
        self.move_particles_by(measurements[PF_DELTA_POS], measurements[PF_DELTA_YAW], measurements[PF_TRACKER_STATUS],
                               self.check_wall_crossing)
        if observations[2] is not None:
            self.score_particles(observations)
            self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)
            self.resample_particles()
            self.tot_motion = 0.
        else:
            self.vis.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

        self.tot_motion += np.linalg.norm(measurements[PF_DELTA_POS], ord=2)
        if self.tot_motion >= 1.5 or len(self.particles)/self.num_particles < 0.25:
            self.resample_particles()
            self.tot_motion = 0.

    def remove_non_walkable_locations(self, x_vector, z_vector):
        tmp_pos = np.column_stack((x_vector, z_vector))
        tmp_pt = self.annotated_map.uv2pixels_vectorized(tmp_pos)

        idx = [self.walkable_mask[np.array(tmp_pt[:, PF_V]), np.array(tmp_pt[:, PF_U])] > 0]
        sample_x = x_vector[idx]
        sample_z = z_vector[idx]

        num_valid = len(sample_x)

        return sample_x, sample_z, num_valid

    def move_particles_by(self, delta_pos, vio_yaw_delta, tracker_status, check_wall_crossing=True):
        start_pt = (self.annotated_map.uv2pixels_vectorized(self.particles[:, PF_X:PF_YAW]))

        self.particles[:, PF_YAW_OFFSET] += np.random.normal(0, 0.01, len(self.particles))

        rotated_delta_pos = [delta_pos[PF_X] * np.cos(self.particles[:,PF_YAW_OFFSET]) + -delta_pos[PF_Z] * np.sin(self.particles[:,PF_YAW_OFFSET]),
                             delta_pos[PF_X] * -np.sin(self.particles[:,PF_YAW_OFFSET]) + -delta_pos[PF_Z] * np.cos(self.particles[:,PF_YAW_OFFSET])]

        n = rotated_delta_pos / (np.linalg.norm(rotated_delta_pos, ord=2) + 1e-6)
        n_hat = [n[1], -n[0]]

        eps1 = np.random.uniform(-np.linalg.norm(rotated_delta_pos, ord=2) * self.position_noise_maj,
                                 np.linalg.norm(rotated_delta_pos, ord=2) * self.position_noise_maj,len(self.particles))

        eps2 = np.random.uniform(-np.linalg.norm(rotated_delta_pos, ord=2)*self.position_noise_min,
                                 np.linalg.norm(rotated_delta_pos, ord=2)*self.position_noise_min, len(self.particles))

        noise = [eps1 * self.particles[:, PF_FUDGE] * n[0], eps1* self.particles[:, PF_FUDGE] * n[1]] + [eps2 * n_hat[0], eps2 * n_hat[1]]
        x = self.particles[:, PF_X] + rotated_delta_pos[PF_X] + noise[PF_X]
        z = self.particles[:, PF_Z] + rotated_delta_pos[PF_Z] + noise[PF_Z]

        dest_pt = self.annotated_map.uv2pixels_vectorized(np.column_stack((x, z)))

        if check_wall_crossing:
            self.particles = set_wall_hit_score(self.particles, start_pt, dest_pt, self.walls_image)
            x = x[self.particles[:, PF_SCORE] >= 0]
            z = z[self.particles[:, PF_SCORE] >= 0]
            self.particles = self.particles[self.particles[:, PF_SCORE] >= 0]

        self.particles[:, PF_X] = x + 0.
        self.particles[:, PF_Z] = z + 0.
        self.particles[:, PF_YAW] += vio_yaw_delta + np.random.normal(0, self.yaw_noise, len(self.particles))
        # print(self.particles)

    def score_particles(self, observations):
        # print(observations[2])
        if observations[2] is not None:
            app = camera_info.get_camera_angle_per_pixel()
            # print("APP:", app)
            start_pt = self.annotated_map.uv2pixels_vectorized(self.particles[:, PF_X:PF_YAW])
            cnt = 0
            yaw_score = np.zeros((len(self.particles), len(self.annotated_map.map_landmarks_dict['exit_sign'])), dtype=np.float64)
            for s in self.annotated_map.map_landmarks_dict['exit_sign']:
                end_pt = self.annotated_map.uv2pixels(s.position)
                ys = np.zeros(len(self.particles), dtype=np.float64)
                # yaw_score[:, cnt]\

                yaw_score[:, cnt] = score_particle_yaw_to_sign(start_pt.astype(np.float64), end_pt.astype(np.float64),
                                               self.particles[:, PF_X:PF_YAW].astype(np.float64), np.asarray(s.position, dtype=np.float64),
                                               self.particles[:, PF_YAW].astype(np.float64),
                                               np.asarray(s.normal, dtype=np.float64), ys, self.walls_image, observations[3],
                                                   float(app))
                tmp_d = np.abs(observations[2] - cdist(self.particles[:, 0:2], [s.position], metric='euclidean'))
                yaw_score[:, cnt] *= 1/(1+.1*tmp_d).squeeze()

                # d_global[:,cnt] = tmp_d.squeeze()
                # d_global[ys == 0, cnt] = 1e6
                cnt += 1
            # d = observations[2] - d_global.min(axis=1)
            # i_d = d_global.argmin(axis=1)
            # d = d_global.min(axis=1)
            # print(d)
            # d = cdist(self.particles[:, 0:2], [observations[0]], metric='euclidean')
            max_score = yaw_score.max(axis=1)
            self.particles[:, PF_SCORE] = max_score + 1e-6

            # self.particles[self.particles[:,PF_SCORE]>=0., PF_SCORE] = (1/(1+.1*d[self.particles[:, PF_SCORE] >= 0.])).transpose()
            # self.particles[self.particles[:, PF_SCORE] >= 0., PF_SCORE] *= yaw_score[np.arange(len(yaw_score)), i_d]
            # self.particles[self.particles[:, PF_SCORE] >= 0., PF_SCORE] += 1e-6 # to avoid crash during resample

    def resample_particles(self):
        tot_score = np.sum(self.particles[:, PF_SCORE])
        w = self.particles[:, PF_SCORE] + 0
        # print(len(self.particles))
        if tot_score > 0:
            w /= tot_score
        else:
            print("!!! Total Particles Score is ZERO")
        idx = np.random.choice(len(w), self.num_particles, p=w)
        new_particles = self.particles[idx]
        # old_scores = self.particles[idx, PF_SCORE]
        self.particles = new_particles
        self.particles[:, PF_SCORE] = 1.
        # self.particles[:, PF_SCORE] = old_scores


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


@jit(nopython=True)
def score_particle_yaw_to_sign(uv_pt1_list, uv_pt2, xy_pt1_list, xy_pt2, yaws, sign_normal, yaw_score, map, sign_roi, angle_per_pixel):
    """Traversability calculation:
    Inputs are pixel locations (r1,c1) and (r2,c2) and 2D map array.
    Draw a line from (r1,c1) to (r2,c2) and determine whether any white (wall) pixels are hit along the way.
    Return 1 if a wall is hit, 0 otherwise."""

    for p in range(0, len(uv_pt1_list)):
        yaw_score[p] = 0.1

        # check particle yaw compatible with sign orientation
        yaw_diff = np.dot(sign_normal, np.array([cos(yaws[p]), sin(yaws[p])]))
        if yaw_diff > -0.2:
            yaw_score[p] = 0.1

        if yaw_score[p] > 0:

            theta_pred = atan2(xy_pt2[1] - xy_pt1_list[p,1], xy_pt2[0] - xy_pt1_list[p,0])
            column_detection = int(sign_roi[0] + sign_roi[2]/2)
            theta_d = yaws[p] + (180 - column_detection) * angle_per_pixel
            x = abs(sin(theta_pred - theta_d))
            yaw_score[p] += 0.2/(0.2+x)

            r1 = int(uv_pt1_list[p][1])
            c1 = int(uv_pt1_list[p][0])
            r2 = int(uv_pt2[1])
            c2 = int(uv_pt2[0])

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
                if map[r, c] > 0:  # hit!
                    yaw_score[p] = 0.1
                    break

    return yaw_score
