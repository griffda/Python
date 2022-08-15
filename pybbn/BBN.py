#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 13:58:57 2022

@author: tomgriffiths

In this script we will attepmt to create a BN using the library pyBBN 
for the BOS model 
"""
import pandas as pd
from pybbn.graph.factory import Factory

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output.csv')
structure = {
    'm': [],
    'r': [],
    'mu': [],
    'theta': [],
    'l': [],
    'KE': ['m', 'r', 'mu', 'theta', 'l']
    }
bbn = Factory.from_data(structure, df)
