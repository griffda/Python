#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 12:58:05 2022

@author: tomgriffiths
THIS IS A SCRIPT FOR MAKING A BAYESIAN NETWORK FOR THE BOS MODEL
"""
from pgmpy.models import BayesianNetwork
import pandas as pd 
from pgmpy.estimators import MaximumLikelihoodEstimator
import numpy as np

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
model.nodes()

model.get_cpds("KE")

#Fitting the model using maximum likelihood estimator
mle = MaximumLikelihoodEstimator(model=model, data=train_data)

#estimating the CPD for a single node
print(mle.estimate_cpd(node="KE"))

#estimating CPDs for all the nodes in the model
mle.get_parameters()[:10] #shows first 10 CPDs in the output 

#verifiying that the learned parameters are almost equal
np.allclose(model.get_cpds('KE'), mle.estimate_cpd('KE'), atol=0.01)

