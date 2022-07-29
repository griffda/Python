#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 11:01:59 2022

@author: tomgriffiths
"""

from pyDOE import lhs
from scipy.stats.distributions import norm
import pandas as pd
import numpy as np

#generating one column of 5 samples for mass around value of 2 with SD of 1
mass = lhs(1, samples=5, criterion=('center'))
mass = norm(loc=2, scale=1).ppf(mass)
#converting array of inputs into dataframe
# dfmass = pd.DataFrame(mass, columns=['A'])
# dfmass.A = dfmass.A.round(2)
# print(dfmass)


#generating 5 samples for radius around value of 1 with SD of 1
radius = lhs(1, samples=5)
radius = norm(loc=1, scale=1).ppf(radius)
#converting array of inputs into dataframe
# dfrad = pd.DataFrame(radius, columns=['B'])  
# # dfrad.B = dfrad.B.round()
# dfrad.B = dfrad.B.astype(int)

#generating 5 samples for mu around value of 0.001 with SD of 0.00015
fric_coeff = lhs(1, samples=5)
fric_coeff = norm(loc=0.001, scale=0.00025).ppf(fric_coeff)
#converting array of inputs into dataframe ### 
# dfmu = pd.DataFrame(mu, columns=['C'])
  

#generating 5 samples for length around value of 10 with SD of 3.5
length = lhs(1, samples=5)
length = norm(loc=10, scale=3.5).ppf(length)
#converting array of inputs into dataframe
# dflen = pd.DataFrame(length, columns=['D'])
# dflen.D = dflen.D.round(2)


#generating 5 samples for theta around value of 0.7 with SD of 0.15
s_angle = lhs(1, samples=5)
s_angle = norm(loc=0.7, scale=0.15).ppf(s_angle)
#converting array of inputs into dataframe
# dftheta = pd.DataFrame(theta, columns=['E'])
# dftheta.E = dftheta.astype(int)

#concatinate into one large dataframe of input values
# mergeddf = [dfmass, dfrad, dfmu, dflen, dftheta]
# inputdf = pd.concat(mergeddf, axis=1)
# inputdf.B = inputdf.B.round(2)
# inputdf = inputdf.astype(float)
# result = inputdf.dtypes
# print(inputdf)

# df = pd.DataFrame() #creates a new dataframe that's empty
# df = pd.concat([inputdf], axis=1, ignore_index=False) # ignoring index is optional

#creating a for loop for bos function 
 
import vpython as vp

outputdf = pd.DataFrame()

# #f=ma model function
def bos():
    ##INPUTS
    # m = inputdf[inputdf.columns[0]]
    m = mass        
    #radius of ball  
    # R = inputdf[inputdf.columns[1]]
    R = radius
    #friction co-efficient
    # mu = inputdf[inputdf.columns[2]]
    mu = fric_coeff
    #acceleration due to gravity
    #g = vp.vector(0,-9.81,0)
    g = -9.81
    #length of the slope
    # L=inputdf[inputdf.columns[3]]
    L = length
    #slope angle
    # theta1 = np.pi/inputdf[inputdf.columns[4]]
    # theta1 = inputdf[inputdf.columns[4]] * (180/np.pi)
    theta1 = s_angle * (180/np.pi)
    # Ll =  inputdf[inputdf.columns[4]]*np.cos(theta1)
    # Lh =  inputdf[inputdf.columns[4]]*np.sin(theta1)
    Ll = s_angle*np.cos(theta1)
    Lh = s_angle*np.sin(theta1)
    #angles for resolving
    # theta = inputdf[inputdf.columns[4]] * (180/np.pi)
    theta = s_angle * (180/np.pi)
    # initial time
    t = 0
    dt = 0.005
    #final time
    T = 2
    
    #positioning the ball and the slope together using vector quantities
    #initial_position = vp.vector(Ll, Lh, 0)
    initial_velocity = vp.vector(0, 0, 0)
    # initial_acc = vp.vector(0, 0, 0)
    initial_position = vp.vector(-L+Ll,-R+Lh, 0)
    #initial_position = vp.vector(8, 17, 0)
    slope_pos = vp.vector(Ll-R, -Lh-R, 0)
    #slope_pos = vp.vector(17, -18, 0)
    slope_angle = vp.vector(Ll, -Lh, 0)
    #slope_angle = vp.vector(18, -18, 0)
    slope_orientation = vp.vector(0,0,0)
    #slope_size = vp.vector(3*10, 2*1, 1)
    slope_size = vp.vector(3*L, 2*R, 1)

    #giving the ball some parameter in space
    ball = vp.sphere(
        pos=initial_position,
        radius=R,
        color=vp.color.cyan,
        make_trail=True,
        # retain=50
    )

    #giving the slope some parameters in space
    slope = vp.box(
        pos=slope_pos, 
        axis=slope_angle, 
        size=slope_size,
        #up=slope_orientation
    )
    
    #giving the ball some conditions
    ball_v = initial_velocity
    #ball_acc = initial_acc
    
    scene = vp.canvas(title = "Ball on slope model")
    # scene = vp.canvas(visualize=False)
    
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
    #g1 = vp.graph(title="Ball on a slope", xtitle="t [s]", ytitle="v [m/s]",width=500, height=250)
    acc = vp.gcurve(color=vp.color.blue, label="a")
    #kinetic = vp.gcurve(color=vp.color.red, label="KE")
    vel = vp.gcurve(color=vp.color.green, label="vel")
    #running the visualisation
    while t < T:
        vp.rate(100)
        # vp.scene.visible = False
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
    ###Creating a new data frame for the outputs
    output = {'output parameters': ['distance', 'final volecity', 'final accelleration', 'final kinetic energy'],
              'symbol': ['d', 'v_f', 'a_f', 'ke_f'],
              'value': [ball.pos.mag, ball_v.x, ball_v.x, k_e],
              'units': ['m', 'm/s', 'm/s^2', 'J']
              }

# user-defined function
# def subtractData(x):
#     return x + 2


# for column in outputdf:
#     bos()
#     output2df = pd.concat([inputdf,outputdf], axis=0, ignore_index=True)
# print(output2df)

#output2df = pd.DataFrame()

# for column in inputdf:
#     subtractData()
#     output2df = pd.concat([inputdf,outputdf], axis=1, ignore_index=True)
# print(output2df)

# outputdf = inputdf.apply(bos)
# output2df = pd.concat([inputdf,outputdf], axis=0, ignore_index=True)
# print(output2df)



    