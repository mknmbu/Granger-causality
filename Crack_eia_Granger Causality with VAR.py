#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import datetime as datetime


# In[2]:


##importing data file
## Importing data file
df = pd.read_csv('Crack_eia.csv',index_col='Date', parse_dates=True)
df=df.dropna()
print('Shape of data',df.shape)
df.head()


# In[3]:


print (df.info())


# In[4]:


from statsmodels.tsa.stattools import grangercausalitytests


# In[5]:


from statsmodels.tsa.stattools import adfuller


# In[6]:


# Stationary test of WTI crude oil
print("Observations of Dickey-fuller test for WTI Crude oil")
x = df["wti"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[7]:


# create new variables
df[ 'lcrack'] = np.log(df['crack'])
df['lwti' ] = np.log(df['wti'])
df['lgas' ] = np.log(df['gasoline'])
df['lheat' ] = np.log(df['heatoil'])


# In[8]:


#
# non-stationary - use first difference
df['dcrack'] = df.crack.diff(1)
df['dlcrack'] = df.lcrack.diff(1)
df['dwti'] = df.wti.diff(1)
df['dlwti'] = df.lwti.diff(1)


# In[10]:


df=df.dropna()


# In[13]:


# Stationary test of WTI crude oil after first differencing
print("Observations of Dickey-fuller test for WTI Crude oil after first differencing")
x = df["dlwti"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[12]:


# Stationary test of Crack spread
print("Observations of Dickey-fuller test")
x = df["crack"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[14]:


# Stationary test of Crack spread after first differencing
print("Observations of Dickey-fuller test after first differencing")
x = df["dlcrack"].values
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('P-value: %f' % result[1])
print('Critical values:')
for key, value in result [4].items():
    print('\t%s: %.3f' % (key,value))
    
if result[0]<result[4]["5%"]:
        print ("Reject Ho - Time series is Stationary")
else:
        print ("Failed to reject Ho - Time series is Non-stationary")


# In[16]:


#perform Granger-Causality test for normal price
grangercausalitytests(df[['crack','wti']], 4, addconst=True, verbose=True)


# In[15]:


#perform Granger-Causality test for log versions of WTI crude oil and Crack spread
grangercausalitytests(df[['lwti', 'lcrack']], 4, addconst=True, verbose=True)


# In[17]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from tqdm import tqdm_notebook
from itertools import product

import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# In[19]:


df_var = pd.read_csv('Crack_eia.csv',index_col='Date', parse_dates=True)
print(df_var.shape)
df_var.head()


# In[21]:


df_var['lwti' ] = np.log(df_var['wti'])
df_var['lcrack' ] = np.log(df_var['crack'])
df_var=df_var.dropna()


# In[22]:


df_var = df_var[['lwti','lcrack']]
print(df_var.shape)


# In[24]:


train_df=df_var[:-1563]
test_df=df_var[-400:]
print(test_df.shape, train_df.shape)


# In[25]:


model = VAR(train_df.diff()[1:])


# In[26]:


sorted_order=model.select_order(maxlags=20)
print(sorted_order.summary())


# In[27]:


var_model = VARMAX(train_df, order=(4,0),enforce_stationarity= True)
fitted_model = var_model.fit(disp=False)
print(fitted_model.summary())


# In[44]:


n_forecast = 200
predict = fitted_model.get_prediction(start=len(train_df),end=len(train_df) + n_forecast-10)#start="2013-01-04",end='2020-12-31')

predictions=predict.predicted_mean


# In[45]:


predictions.columns=['lwti_predicted','lcrack_predicted']
print (predictions)


# In[46]:


test_vs_pred=pd.concat([test_df,predictions],axis=1)


# In[47]:


test_vs_pred.plot(figsize=(13,7))


# In[ ]:




