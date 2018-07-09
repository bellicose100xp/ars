# -*- coding: utf-8 -*-
import os
import numpy as np

# define hyper parameters
class Hp():
    def __init__(self):
            self.nb_steps = 1000
            self.episode_length = 1000
            self.learning_rate = 0.02
            self.nb_directions = 16
            self.nb_best_directions = 16
            # adding test assertions to detect inconsistancies
            assert self.nb_best_directions <= self.nb_directions
            self.noise = 0.03
            self.seed = 1
            self.env_name = ''

# normalizing the states
class Normalizer():
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        # numerator of mean
        self.mean_diff = np.zeros(nb_inputs)
        # variance
        self.var = np.zeros(nb_inputs)
        
    def observe(self, x):
        self.n += 1. # dot at the end so it's treated as floating point
        # keeping a copy of the mean for future calculations
        last_mean = self.mean.copy() # shallow copy
        # computing new online mean
        self.mean += (x - self.mean) / self.n
        # computing new online numerator
        self.mean_diff += (x - last_mean) * (x - self.mean)
        # since we divide by variance, it can never be zero
        # this is done by adding a .clip() to assigna min value
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        # observed standard deviation
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI

        
        