#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:01:59 2022

@author: tomgriffiths
"""

# from pyDOE import lhs
# from scipy.stats.distributions import norm
import pandas as pd
import numpy as np
import vpython as vp


# #generating one column of 5 samples for mass around value of 2 with SD of 1
# mass = lhs(1, samples=5, criterion=('center'))
# sample_masses = norm(loc=2, scale=1).ppf(mass)


# #generating 5 samples for radius around value of 1 with SD of 1
# radius = lhs(1, samples=5)
# sample_radii = norm(loc=1, scale=1).ppf(radius)

# #generating 5 samples for mu around value of 0.001 with SD of 0.00015
# fric_coeff = lhs(1, samples=5)
# sample_fric_coeffs = norm(loc=0.001, scale=0.00025).ppf(fric_coeff)
  

# #generating 5 samples for length around value of 10 with SD of 3.5
# length = lhs(1, samples=5)
# sample_lengths = norm(loc=10, scale=3.5).ppf(length)


# #generating 5 samples for theta around value of 0.7 with SD of 0.15
# s_angle = lhs(1, samples=5)
# sample_s_angles = norm(loc=0.7, scale=0.15).ppf(s_angle)


# # Stack the samples into a 2D array so we can loop through all 5 one by one.
# sample_array = np.column_stack((sample_masses, sample_radii, sample_fric_coeffs, sample_lengths, sample_s_angles))

def ball_on_slope(sample_array):
    """
    This function is expecting a 2D numpy array. Each row in the array will contain 5 values:
    [mass, radius, length, friction coefficient, s angle]. 

    We will run the ball on a slope visualisation for each row of sample values provided.
    """
    #Creating an array filled with zeros with the shape of "sample_array"
    output_array=np.zeros((np.shape(sample_array)[0],3))
    #Loop that runs in parallel(?) over "sample_array" 
    for i in range(np.shape(sample_array)[0]):
        m, R, mu, L, s_angle, Ll, Lh, theta1, theta, Ff, F_norm, F_norm_v, F_f_v, w = get_initial_parameters(sample_array[i,:])

        (
            initial_velocity,
            initial_acc,
            initial_position,
            slope_pos, slope_angle,
            slope_orientation,
            slope_size
        ) = set_ball_and_slope_positioning(L, Ll, Lh, R)
        

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


        ##THESE ARE SOME LINES FOR CREATING GRAPHS
        #scene = vp.canvas(title = "Ball on slope model"
        #g1 = vp.graph(title="Ball on a slope", xtitle="t [s]", ytitle="v [m/s]",width=500, height=250)
        #acc = vp.gcurve(color=vp.color.blue, label="a")
        #kinetic = vp.gcurve(color=vp.color.red, label="KE")
        #vel = vp.gcurve(color=vp.color.green, label="vel")
     
        ##This runs a function using some parameters, and then returning a tuple containing slope, ball, ball_v and k_e
        ball_a, ball, ball_v, k_e = run_visualisation(F_norm, F_norm_v, F_f_v, Ff, m, ball_v, ball, theta1, theta)
        
        ##Vectorised (?) loop for generating an output array of ball_v, ball_a, and k_e. 
        output_array[i,:] = [ball_v, ball_a, k_e]
       
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
                           columns = ['m', 'r', 'mu', 'theta', 'l'],
                           )
    
     out_dat = pd.DataFrame(output_array[:,:], 
     columns = ['vf', 'af', 'KE'],
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
    m, R, mu, L, s_angle = samples
    g = -9.81
    # slope angle
    theta1 = s_angle * (180/np.pi)
    Ll = s_angle*np.cos(theta1)
    Lh = s_angle*np.sin(theta1)
    #angles for resolving
    theta = s_angle * (180/np.pi)
    #horizontal component friction
    Ff = m * g * np.cos(theta)**2 * mu
    #horizontal component of normal force
    F_norm = m * g * np.cos(theta) * np.sin(theta)

    #vertical component of normal force
    F_norm_v = m * g * np.cos(theta) * np.cos(theta)
    #vertical component friction friction force
    F_f_v = mu * m * g * np.cos(theta) * np.sin(theta)
    #weight of ball itself
    w = m * g 
    #anglular momentum
    #W = 0

    return m, R, mu, L, s_angle, Ll, Lh, theta, theta1, Ff, F_norm, F_norm_v, F_f_v, w


def set_ball_and_slope_positioning(L, Ll, Lh, R):
    initial_velocity = vp.vector(0, 0, 0)
    initial_acc = vp.vector(0, 0, 0)
    initial_position = vp.vector(-L+Ll,-R+Lh, 0)
    slope_pos = vp.vector(Ll-R, -Lh-R, 0)
    slope_angle = vp.vector(Ll, -Lh, 0)
    slope_orientation = vp.vector(0,0,0)
    slope_size = vp.vector(3*L, 2*R, 1)

    return initial_velocity, initial_acc, initial_position, slope_pos, slope_angle, slope_orientation, slope_size


def run_visualisation(F_norm, F_norm_v, F_f_v, Ff, m, ball_v, ball, theta1, theta):
    t = 0
    dt = 0.005
    #final time
    T = 0.5
    g = -9.81
    w = m * g
    while t < T:
        #vp.rate(100)
        vp.scene.visible = False
        # vp.scene.width = scene.height = 1
        #if -w*R>=ball_v.mag/m:
            #Ff = vp.vector(0,0,0)
        ball_net = -F_norm + Ff
        ball_v.x = ball_v.x + ((ball_net) / m) * dt
        ball_v.y = ball_v.y + ((w - F_norm_v - F_f_v) / m) * dt
        ball.pos = ball.pos + ball_v * dt
        ball_a = (2/3) * g * np.sin(theta)
        ball_a += ball_v.mag * dt
        k_e = 0.5 * m * ball_v.mag**2
        #kinetic.plot(t, k_e)
        #acc.plot(t, ball_a)
        #vel.plot(t, ball_v.mag)
        t += dt
    
    return ball.pos.mag, ball_v, k_e, ball_a