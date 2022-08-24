#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 12:33:36 2022

@author: tomgriffiths
"""

import numpy as np
from pyDOE import lhs
from scipy.stats.distributions import norm
from bos_sampling import ball_on_slope
from bos_sampling import create_output_dataframe

#generating one column of 5 samples for mass around value of 2 with SD of 1
mass = lhs(1, samples=200)
sample_masses = norm(loc=1, scale=0.01).ppf(mass)

###generating 5 samples for radius around value of 1 with SD of 1
###Can add this as a sample later - perhaps as a method of testing what happens when we get new data
v0 = lhs(1, samples=200)
sample_v0 = norm(loc=1, scale=0.15).ppf(v0)

#generating 5 samples for theta around value of 0.7 with SD of 0.15
s_angle = lhs(1, samples=200)
sample_s_angles = norm(loc=3, scale=0.15).ppf(s_angle)

##This runs a function using some parameters, and then returning a tuple containing sample array
sample_array = np.column_stack((sample_masses, sample_v0, sample_s_angles))

#This runs a function using some parameters, and then returning a tuple containing output array
output_array = ball_on_slope(sample_array)
###FUNCTIONS 
#ball_on_slope(sample_array)
create_output_dataframe(output_array, sample_array)


