import pandas as pd
import numpy as np

df = pd.read_csv('/Users/tomgriffiths/OneDrive - Imperial College London/Research/Python/gitlibraries/Python/output2.csv',
                 index_col=False,
                 usecols=['m', 'theta','v0', 'vf', 'KE'],
                 encoding=('utf-8')
                 )



##Create new data frame - call it binned and fill with the values and then use structure syntax below. 

labels = ['a','b']
df['m_bins'] = pd.qcut(df['m'], 2, labels=labels)
df['theta_bins'] = pd.qcut(df['theta'], 2, labels=labels)
df['v0_bins'] = pd.qcut(df['v0'], 2, labels=labels)
df['vf_bins'] = pd.qcut(df['vf'], 2, labels=labels)
df['KE_bins'] = pd.qcut(df['KE'], 2, labels=labels)

# df['n']= np.digitize(df['m'], np.linspace(df['m'].min(), df['m'].max(), 3))
# df['t']= np.digitize(df['theta'], np.linspace(df['theta'].min(), df['theta'].max(), 3))
# df['v']= np.digitize(df['v0'], np.linspace(df['v0'].min(), df['v0'].max(), 3))
# df['c']= np.digitize(df['vf'], np.linspace(df['vf'].min(), df['vf'].max(), 3))
# df['k']= np.digitize(df['KE'], np.linspace(df['KE'].min(), df['KE'].max(), 3))



df_binned = df.drop(['m', 'theta','v0', 'vf', 'KE'], axis=1)
print(df_binned.head(10))

df_binned.to_csv('binned_data.csv', index=False)