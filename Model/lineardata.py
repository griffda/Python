import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from pyDOE import lhs
from scipy.stats.distributions import norm

###Generate data from linear function with number of data samples n
def make_data(n):

    ###generating one column of n samples for mass around value of 5 with SD of 0.2
    # mass = lhs(1, samples=(n))
    # sample_masses = norm(loc=5, scale=0.2).ppf(mass)
    ###generating one column of n samples for mass around value of 5 with SD of 0.2 using a unfirom distribution
    sample_masses = np.random.uniform(size=n, low=4.8, high=5.2)
    
    ###generating one column of n samples for force around value of 10 with SD of 0.2
    # force = lhs(1, samples=(n))
    # sample_force = norm(loc=10, scale=0.2).ppf(force)
    ###generating one column of n samples for force around value of 10 with SD of 0.2 using a unfirom distribution
    sample_force = np.random.uniform(size=n, low=9.8, high=10.2)

     
     

    ### f = ma rearranged to calculate acceleration given a force and a mass. 
    acc = sample_force / sample_masses 

    ###this collects the data into an array
    sample_array = np.column_stack((sample_masses, sample_force, acc))

    ###This plots the data 
    plt.scatter(sample_masses, acc.ravel())
    plt.title("F = ma", fontweight="bold", size=6)
    plt.ylabel('Acceleration $m s^{-2}$', fontsize=7)  # Y label
    plt.xlabel('Mass (kg)', fontsize=7)  # X label
    
    return sample_array  


def create_output_dataframe(sample_array):
     """
     using ball_on_slope function, this collates all output data into one dataframe and output.csv file 
     """
     ###Creating a new data frame for the outputs
     ###Note: you will get n number of outputs (n=no. of samples), so you'll need to return a datastructure that
     ###contains all of them.
     
     out_dat = pd.DataFrame(sample_array[:,:],
                           columns = ['mass', 'force', 'acceleration'],
                           )
    
         ###Saving to new csv after data manipulation
     out_dat.to_csv('outputv4.csv', index=False)
     return out_dat

###FUNCTIONS 
sample_array = make_data(1000)
out_dat = create_output_dataframe(sample_array)     
# plt.show()