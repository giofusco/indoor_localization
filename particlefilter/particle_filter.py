import numpy as np
from plotting.visualizer import Visualizer

class ParticleFilter:

    # initialize particle filter from configuration file
    def __init__(self, config_file):
        self.particles = []
        raise NotImplementedError

    def __init__(self, annotated_map, num_particles=1000):
        self.num_particles = num_particles
        self.particles = []
        self.annotated_map = annotated_map

#todo: check if particle is outside of walkable area
    def initialize_particles_at(self, pos, yaw, position_noise_sigma=0.1, yaw_noise_sigma=0.1):
        #initialize N random particles all over the walkable area
        sample_x = np.random.normal(pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(pos[1], position_noise_sigma, self.num_particles)
        yaws = np.random.normal(yaw, yaw_noise_sigma, self.num_particles)
        weights = np.zeros((self.num_particles))
        self.particles = np.column_stack((sample_x, sample_y, yaws, weights))
        Visualizer.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def step(self, measurements, observations):
        print ("Particle Filter step")
        self.move_particles_by(measurements[0], measurements[1], position_noise_sigma=0.1)
        Visualizer.plot_particles(annotated_map=self.annotated_map, particles=self.particles)

    def move_particles_by(self, delta_pos, delta_yaw, position_noise_sigma=0.1):
        sample_x = np.random.normal(delta_pos[0], position_noise_sigma, self.num_particles)
        sample_y = np.random.normal(delta_pos[1], position_noise_sigma, self.num_particles)
        self.particles[:,0] += sample_x
        self.particles[:, 1] += sample_y
        Visualizer.plot_particles(annotated_map=self.annotated_map, particles=self.particles)