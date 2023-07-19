import pandas as pd
import numpy as np


df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv', 
                usecols=['m', 'theta','v0', 'vf', 'KE'])

#print(df.head(10))

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 


df['m_bins']= np.digitize(df['m'], np.linspace(df['m'].min(), df['m'].max(), 10))
df['theta_bins']= np.digitize(df['theta'], np.linspace(df['theta'].min(), df['theta'].max(), 10))
df['v0_bins']= np.digitize(df['v0'], np.linspace(df['v0'].min(), df['v0'].max(), 10))
df['vf_bins']= np.digitize(df['vf'], np.linspace(df['vf'].min(), df['vf'].max(), 10))
df['KE_bins']= np.digitize(df['KE'], np.linspace(df['KE'].min(), df['KE'].max(), 10))

df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)

print(df_binned.head(10))

df_binned.to_csv('output2_.csv', index=True)