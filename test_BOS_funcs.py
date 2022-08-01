#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:33:36 2022

@author: tomgriffiths
"""

import numpy as np
from bos_sampling import ball_on_slope
#from bos_sampling import create_output_dataframe
from pyDOE import lhs
from scipy.stats.distributions import norm

#generating one column of 5 samples for mass around value of 2 with SD of 1
mass = lhs(1, samples=5, criterion=('center'))
sample_masses = norm(loc=2, scale=1).ppf(mass)


#generating 5 samples for radius around value of 1 with SD of 1
radius = lhs(1, samples=5)
sample_radii = norm(loc=1, scale=1).ppf(radius)

#generating 5 samples for mu around value of 0.001 with SD of 0.00015
fric_coeff = lhs(1, samples=5)
sample_fric_coeffs = norm(loc=0.001, scale=0.00025).ppf(fric_coeff)
  

#generating 5 samples for length around value of 10 with SD of 3.5
length = lhs(1, samples=5)
sample_lengths = norm(loc=10, scale=3.5).ppf(length)


#generating 5 samples for theta around value of 0.7 with SD of 0.15
s_angle = lhs(1, samples=5)
sample_s_angles = norm(loc=0.7, scale=0.15).ppf(s_angle)

sample_array = np.column_stack((sample_masses, sample_radii, sample_fric_coeffs, sample_lengths, sample_s_angles))

#output_array, ball, ball_v, ball_a, k_e = ball_on_slope(sample_array)

###FUNCTIONS 
ball_on_slope(sample_array)
#create_output_dataframe(output_array)
# set_ball_and_slope_positioning(L, Ll, Lh, R)
# run_visualisation(F_norm, F_norm_v, F_f_v, Ff, m, ball_v, ball, theta1, theta)