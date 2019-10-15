#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[5]:


from scipy.stats import norm
import scipy.stats as stats
flatmiddle = pd.read_csv("artificialWithAnomaly/art_daily_flatmiddle.csv")
flatmiddle.set_index("timestamp", inplace = True)


# In[6]:


def p_value(start_time, end_time, X, alpha):
    '''
    param start_time: start time for the window,  timestamp format
    param end_time: end time for the window,  timestamp format
    param X: dataframe for time series data
    Output: the p-value of each data point in the window and the dataframe for rejected data
            
    '''
    data_window = np.array(X[start_time: end_time].value)
    mean, std = np.mean(data_window), np.std(data_window)
    z_scores = (data_window-mean)/std
    p_value = 1-norm.cdf(abs(z_scores))
    reject = X[start_time: end_time].loc[p_value< alpha]
    return p_value, reject
    
    


# In[21]:


p, reject = p_value('2014-04-10 00:00:00', '2014-04-12 00:00:00', flatmiddle, alpha = 0.05)


# In[12]:


ax = flatmiddle.plot(figsize = (20,10))


# In[22]:


reject


# In[ ]:




