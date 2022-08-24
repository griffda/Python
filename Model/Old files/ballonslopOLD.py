#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 12:21:55 2022

@author: tomgriffiths
"""

import vpython as vp
import numpy as np
import csv
   

#input parameters 
#mass 
m = 2.97
#radius of ball  
R = 1
#x-direction velocity
v0 = 2
#friction co-efficient
mu = 0.0001
#acceleration due to gravity
#g = vp.vector(0,-9.81,0)
g = -9.81
#length of the slope
L=10
#angles for resolving
theta = np.pi/18
#slope angle
theta1 = np.pi/18
Ll = L*np.cos(theta1)
Lh = L*np.sin(theta1)
v0_x = v0*np.cos(theta1)
v0_y = v0*np.sin(theta1)

# initial time
t = 0 
dt = 0.005
#final time
T = 5 
#angluar velocity
I = (2/5) * m * R**2
#angular mom
w = 0 
a_initial = 0
#acc solved
a = (2/3) * g * np.sin(theta)

#positioning the ball and the slope together using vector quantities
#initial_position = vp.vector(Ll, Lh, 0)
initial_velocity = vp.vector(v0_x, -v0_y, 0)
initial_acc = vp.vector(0, 0, 0)

initial_position = vp.vector(-L+Ll,-R+Lh, 0)
slope_pos = vp.vector(Ll-R, -Lh-R, 0)
slope_angle = vp.vector(Ll, -Lh, 0)
#slope_orientation = vp.vector(0,0,0)
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
W = 0

# A plot
g1 = vp.graph(title="Ball on a slope", xtitle="t [s]", ytitle="v [m/s]",width=500, height=250)
acc = vp.gcurve(color=vp.color.blue, label="a")
kinetic = vp.gcurve(color=vp.color.red, label="KE")
vel = vp.gcurve(color=vp.color.green, label="vel")

# #running the visualisation
# while t < T:
#     vp.rate(100)
#     #if -w*R>=ball_v.mag/m:
#         #Ff = vp.vector(0,0,0)
#     ball_net = -F_norm + Ff
#     ball_v.x = ball_v.x + ((ball_net) / m) * dt
#     ball_v.y = ball_v.y + ((w - F_norm_v - F_f_v) / m) * dt
#     ball.pos = ball.pos + ball_v * dt
#     ball_a = (2/3) * g * np.sin(theta)
#     ball_v.mag += ball_a * dt
#     k_e = 0.5 * m * ball_v.mag**2
#     #kinetic.plot(t, k_e)
#     acc.plot(t, ball_a)
#     vel.plot(t, ball_v.mag)
#     t += dt

#running the visualisation
while t < T:
    vp.rate(100)
    #if -w*R>=ball_v.mag/m:
        #Ff = vp.vector(0,0,0)
    #ball_net = -F_norm + Ff
    #ball_v.x = ball_v.x + ((ball_net) / m) * dt
    #ball_v.y = ball_v.y + ((w - F_norm_v - F_f_v) / m) * dt
    #ball.pos = ball.pos + ball_v * dt
    ball_a = (2/3) * g * np.sin(theta)
    ball_v.x -= ball_a * dt * np.cos(theta)
    ball_v.y += ball_a * dt * np.sin(theta)
    ball.pos = ball.pos + ball_v * dt
    k_e = 0.5 * m * ball_v.mag**2
    acc.plot(t, ball_a)
    vel.plot(t, ball_v.mag)
    t += dt
    

print("d = ",ball.pos.mag," m")
print("v final = ",ball_v.mag," m/s")   
print("final accelleration = ", ball_a, "m/s**2") 
print("KE final = ",k_e, " J")

values = [ball.pos.mag, ball_v.x, ball_a, k_e]
value_unit = [ 'm', 'm/s', 'm/s**2', 'J']
    
# with open('output.csv', 'w', newline='') as csvfile:
#     fieldnames = ['unit', 'value']
#     thewriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
#     thewriter.writeheader()
#     for value in values :
#         thewriter.writerow({'value':value, 'unit':value_unit})

        
