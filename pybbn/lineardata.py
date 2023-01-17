import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from pyDOE import lhs
from scipy.stats.distributions import norm

###Generate data from linear function 

def make_data(n):
    mass = lhs(1, samples=(n))
    sample_masses = norm(loc=5, scale=0.2).ppf(mass)

    force = lhs(1, samples=(n))
    sample_force = norm(loc=10, scale=0.2).ppf(force) 

    acc = sample_force / sample_masses 

    sample_array = np.column_stack((sample_masses, sample_force, acc))

    plt.scatter(sample_force, acc.ravel())
    plt.title("F = ma", fontweight="bold", size=6)
    plt.ylabel('Acceleration $m s^{-2}$', fontsize=7)  # Y label
    plt.xlabel('Force (N)', fontsize=7)  # X label

    return (sample_array)  


def create_output_dataframe(sample_array):
     """
     using ball_on_slope function, this collates all output data into one dataframe and output.csv file 
     """
     ##Creating a new data frame for the outputs
     ##Note: you will get n number of outputs (n=no. of samples), so you'll need to return a datastructure that
     ##contains all of them.
     
     out_dat = pd.DataFrame(sample_array[:,:],
                           columns = ['mass', 'force', 'acceleration'],
                           )
    
    #  out_dat = pd.DataFrame(output_array[:,:], 
    #  columns = [],
    #  )
    
    #  out_dat2 = [in_dat, out_dat]
     
    #  out_dat3 = pd.concat(out_dat2, axis=1, join="inner")

     ###Saving to new csv after data manipulation
     out_dat.to_csv('outputv3.csv', index=False)
     
     return out_dat

###FUNCTIONS 
#ball_on_slope(sample_array)

sample_array = make_data(500)
out_dat = create_output_dataframe(sample_array)     

# acc, sample_masses, sample_force, sample_array, output_array = make_data(500)
# fig, ax = plt.subplots(dpi=100)
# plt.scatter(acc, sample_masses)
# plt.title("F = ma", fontweight="bold", size=6)
# plt.ylabel('Acceleration $m s^{-2}$', fontsize=7)  # Y label
# plt.xlabel('Mass (kg)', fontsize=7)  # X label
plt.show()