#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:58:05 2022

@author: tomgriffiths
THIS IS A SCRIPT FOR MAKING A BAYESIAN NETWORK FOR THE BOS MODEL
"""
from pgmpy.models import BayesianNetwork
import pandas as pd 

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

#Estimates the CPD for each variable based on a given data set
model.fit(train_data)
model.get_cpds()
print(model.get_cpds('KE'))
