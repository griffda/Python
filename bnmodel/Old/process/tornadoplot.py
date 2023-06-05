import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# Load the data into a pandas DataFrame
# df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/PROCESS-/griff_work/HPC/monte_carlo/mc_output_test_uniform/hdf08032023.csv')
df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/PROCESS-/griff_work/HPC/monte_carlo/mc_output_test_uniform/hdf08032023.csv', index_col=False)
df = df.head()
# print(df)
# print(df.values)

# calculate the normalised coe
normalised_df = (df-df.mean())/df.std()
print(normalised_df)
# df['mean_coe'] = df['coe'].mean()
# print(df['mean_coe'])
# df2 = df.to_dict('list')
df2 = normalised_df.to_dict('list')
# print(df)

# sort the data by rel_coe
# data_rel = np.array(list(df.sort_values('rel_coe', ascending=False)))
# data = np.array(list(df.T.values))
# data = np.array(list(normalised_df.T.values[:,:-2]))
data = np.array(list(normalised_df.T.values))

# data = np.array(list(df.values))

# print(data)
# data = np.array(list(df.values))
# for i in df.iloc[:,:].T.values: 
#     print(i)   
#     widths = np.max(i) - np.min(i)
#     print(widths)
    # data = df.iloc[:,:].T.values
    # widths = data[i]
    # print(data[i]) 
# data2 = np.array(df(1))
# print(data2)
# print(data)
# labels = list(df.columns[:-2])
labels = list(normalised_df.columns[:-2])
# widths = max(data[1, :]) - min(data[1, :])

# print(widths)
data_cum = data.cumsum(axis=1)
# print(data_cum)
middle_index = data.shape[1]//2
offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2

# print(middle_index)
# print(offsets)


# create the plot
fig, ax = plt.subplots(figsize=(8, 6))

# print(df2.items())

# plot bars: this plot needs to plot a value for each value of coe, i.e., for every other param there
# a corresponding coe value. 
# loop should go through every param array 

i = 0
count = 0


# for varName, index in df2.items(): 
#     edge = np.zeros((len(df2.items()), len(index[:])))

#     print(varName)
#     print(index)
    
#     for i in range(len(index)):
#         edge[count, i] = index[i]
#     print(edge)
data2 = np.array(list(normalised_df.values[:,:-2]))

data_cum = data2.T.cumsum(axis=1)
# print(data2)

widths = {}

for i, (coe, vals) in enumerate(zip(data[7], data[::,0])):
    # print(i)
    print(coe)
    print(vals)
    # widths[coe] = vals[i]
    # print(widths)
    # widths[count, i] = vals[i]
    # for i2 in vals:
    #     print(i2)
        # widths[count, i2] = vals[i2]
        # print(widths)
    
    # widths = max(data[i, :]) - min(data[i, :])
    # print(widths)
    # starts = data_cum[i, :] - widths - offsets
    # # print(widths)
    # # print(starts)
    # rects = ax.barh(labels, widths, height=0.5, color = "gray")
    
i += 0
count += 0

# set the x-axis limits
ax.set_xlim([-2, 2])
# ax.set_xlim(min(data[7]), max(data[7]))


# plot the relative coe line
ax.axvline(normalised_df['coe'].mean(), color='black', linestyle='--')

# plot the relative coe values
# for i, v in enumerate(df['rel_coe']):
#     ax.text(v + 0.02, i, str(round(v, 2)), color='black', fontweight='bold')

# set the x-axis label
ax.set_xlabel('Cost of Electricity')

# set the y-axis label
ax.set_ylabel('Parameter')

# set the plot title
ax.set_title('Tornado Plot')

# plt.show()
