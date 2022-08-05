#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:58:05 2022

@author: tomgriffiths
THIS IS A SCRIPT FOR MAKING A BAYESIAN NETWORK FOR THE BOS MODEL
"""
from pgmpy.models import BayesianNetwork
from sklearn.linear_model import LinearRegression
#from pgmpy.factors import TabularCPD
import pandas as pd 
from bos_sampling import ball_on_slope
from bos_sampling import create_output_dataframe

#import output csv from BOS_functions script. 
df = pd.read_csv('output.csv', usecols = ['m', 'r', 'mu', 'theta', 'l', 'final velocities', 'final accellerations', 'KE'],
encoding=('utf-8'))

#this is creating a new variable of data from the data frame to train the BN
train_data = df[:10]

##DEFINE MODEL STRUCTURE
#this gives the BN it's structure and shows how each node is related to each other
model = BayesianNetwork([('mu', 'KE'), ('r', 'KE'),
                       ('m', 'KE'), ('theta', 'KE'),
                       ('l', 'KE')])

#using parmeter learning this trains the data
model.fit(train_data)
model.get_cpds()
print(model.get_cpds('KE'))
