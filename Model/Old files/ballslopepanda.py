#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 14:32:20 2022

@author: tomgriffiths
"""

import pandas as pd
import vpython as vp
import numpy as np


###loading in data using CSV
#df = pd.read_csv('input.csv')

###loading in data using excel spreadsheet
df = pd.read_excel('input.xlsx')
#print(df.head(2))

###read headers
#print(df.columns)

###read each column
#print(df['input paramter'][0:2])

###read each row
#print(df.iloc[1:4])
###iterate through each row to access data
#for index, row in df.iterrows():
    #print(index, row[['symbol', 'units'][0:3]])
###find data with specifc word
#print(df.loc[df['units'] == "m"])
###sort values
#print(df.sort_values('value', ascending=True))

###read specific location
#print(df.iloc[2,6])

#########SLOPE PROBLEM

###Defining the lengths of our slope from the input file
Ll = df.iloc[5,2]*np.cos(df.iloc[6,2])
Lh = df.iloc[5,2]*np.sin(df.iloc[6,2])


#positioning the ball and the slope together using vector quantities
initial_velocity = vp.vector(0, 0, 0)
initial_acc = vp.vector(0, 0, 0)

initial_position = vp.vector(-df.iloc[5,2]+Ll,-df.iloc[5,2]+Lh, 0)
slope_pos = vp.vector(Ll-df.iloc[1,2], -Lh-df.iloc[1,2], 0)
slope_angle = vp.vector(Ll, -Lh, 0)
#slope_orientation = vp.vector(0,0,0)
slope_size = vp.vector(3*df.iloc[5,2], 2*df.iloc[1,2], 1)

#giving the ball some parameter in space
ball = vp.sphere(
    pos=initial_position,
    radius=df.iloc[1,2],
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


###Saving to new csv after data manipulation
df.to_csv('output.csv', index=False)
df.to_csv('output.xlsx', index=False)
