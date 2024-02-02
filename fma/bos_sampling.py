#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:01:59 2022

@author: tomgriffiths
"""

# from pyDOE import lhs
# from scipy.stats.distributions import norm
from array import array
import pandas as pd
import numpy as np
import vpython as vp

def ball_on_slope(sample_array):
    """
    This function is expecting a 2D numpy array. Each row in the array will contain 5 values:
    [mass, radius, length, friction coefficient, s angle]. 

    We will run the ball on a slope visualisation for each row of sample values provided.
    """
    #Creating an array filled with zeros with the shape of "sample_array"
    output_array=np.zeros((np.shape(sample_array)[0],2))
    #Loop that runs in parallel(?) over "sample_array" 
    for i in range(np.shape(sample_array)[0]):
        m, v0, R, L, s_angle, Ll, Lh, v0_x, v0_y, theta = get_initial_parameters(sample_array[i,:])

        (
            initial_velocity,
            initial_acc,
            initial_position,
            slope_pos, slope_angle,
            slope_orientation,
            slope_size
        ) = set_ball_and_slope_positioning(v0_x, v0_y, L, Ll, Lh, R)
        

        #giving the ball some parameter in space
        ball = vp.sphere(
            pos=initial_position,
            radius=R,
            color=vp.color.cyan,
            make_trail=False,
            retain=50
        )

        #giving the slope some parameters in space
        slope = vp.box(
            pos=slope_pos, 
            axis=slope_angle, 
            size=slope_size,
            up=slope_orientation
        )

        #giving the ball some conditions
        ball_v = initial_velocity

     
        ##This runs a function using some parameters, and then returning a tuple containing slope, ball, ball_v and k_e
        ball, ball_v, k_e = run_visualisation(m, ball_v, ball, theta)
        ##Vectorised (?) loop for generating an output array of ball_v, ball_a, and k_e. 
        output_array[i,:] = [ball_v, k_e]
    
    return output_array


##OUTPUT DATAFRAME
def create_output_dataframe(output_array, sample_array):
     """
     using ball_on_slope function, this collates all output data into one dataframe and output.csv file 
     """
     ##Creating a new data frame for the outputs
     ##Note: you will get n number of outputs (n=no. of samples), so you'll need to return a datastructure that
     ##contains all of them.
     
     in_dat = pd.DataFrame(sample_array[:,:],
                           columns = ['m', 'theta', 'v0'],
                           )
    
     out_dat = pd.DataFrame(output_array[:,:], 
     columns = ['vf', 'KE'],
     )
    
     out_dat2 = [in_dat, out_dat]
     
     out_dat3 = pd.concat(out_dat2, axis=1, join="inner")
     
     ###Saving to new csv after data manipulation
     out_dat3.to_csv('output2.csv', index=True)
     
     return out_dat

def get_initial_parameters(samples):
    """
    Using the array of samples, calculate and return all the required parameters for the ball on a slope
    visualisation
    """

    # Unpack samples
    m, v0, s_angle = samples
    g = -9.81
    R = 1
    L = 10
    #v0 = 2
    # slope angle
    theta1 = s_angle * (180/np.pi)
    Ll = s_angle*np.cos(theta1)
    Lh = s_angle*np.sin(theta1)
    v0_x = v0*np.cos(theta1)
    v0_y = v0*np.sin(theta1)
    #angles for resolving
    theta = s_angle * (180/np.pi)
    return m, v0, R, L, s_angle, Ll, Lh, v0_x, v0_y, theta


def set_ball_and_slope_positioning(v0_x, v0_y, L, Ll, Lh, R):
    initial_velocity = vp.vector(v0_x, -v0_y, 0)
    initial_acc = vp.vector(0, 0, 0)
    initial_position = vp.vector(-L+Ll,-R+Lh, 0)
    slope_pos = vp.vector(Ll-R, -Lh-R, 0)
    slope_angle = vp.vector(Ll, -Lh, 0)
    slope_orientation = vp.vector(0,0,0)
    slope_size = vp.vector(3*L, 2*R, 1)

    return initial_velocity, initial_acc, initial_position, slope_pos, slope_angle, slope_orientation, slope_size


def run_visualisation(m, ball_v, ball, theta):
    t = 0
    dt = 0.005
    #final time
    T = 5
    g = -9.81
    while t < T:
        #vp.rate(100)
        vp.scene.visible = False
        # vp.scene.width = scene.height = 1
        #if -w*R>=ball_v.mag/m:
            #Ff = vp.vector(0,0,0)
        ball_a = (2/3) * g * np.sin(theta)
        ball_v.x -= ball_a * dt * np.cos(theta)
        ball_v.y += ball_a * dt * np.sin(theta)
        ball.pos = ball.pos + ball_v * dt
        k_e = 0.5 * m * ball_v.mag**2
        t += dt
    return ball.pos.mag, ball_v.mag, k_e

