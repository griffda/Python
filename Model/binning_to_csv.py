import pandas as pd
import numpy as np

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 index_col=False,
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )

##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

labels = [1,2,3,4]
df['m_bins'] = pd.qcut(df['m'], 4, labels=labels)
df['theta_bins'] = pd.qcut(df['theta'], 4, labels=labels)
df['v0_bins'] = pd.qcut(df['v0'], 4, labels=labels)
df['vf_bins'] = pd.qcut(df['vf'], 4, labels=labels)
df['KE_bins'] = pd.qcut(df['KE'], 4, labels=labels)

df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
print(df_binned.head(10))

df_binned.to_csv('binned_data.csv', index=False)