#!/usr/bin/env python
# coding: utf-8

# In[75]:


# import the required libraries
from LinearRegression import LR_1var
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


# ### Load the data

# In[84]:


# read data from csv using pandas
data = pd.read_csv("parkinsons_dataset.csv")


# ### Analyze the data

# In[77]:


data[['PPE','motor_UPDRS']].describe()


# This extracts data of two columns features i:e; PPE and motor_UPDRS from data frame.

# ### Drop null rows

# In[78]:


data = data.dropna(axis=0)


# This helps to remove null values if exists

# ### Data Visualization

# In[79]:


X = data['PPE'].values
Y = data['motor_UPDRS'].values
plt.figure(figsize=(20, 4)) 

plt.subplot(1, 2, 1)
plt.boxplot(X, vert=False)
plt.title('PPE')

plt.subplot(1, 2, 2)
plt.boxplot(Y, vert=False)  
plt.title('motor_UPDRS')

plt.show()


# PPE and motor_UPDRS columns contain outliers, which depicts potential instability in the data. Such outliers have negatively impact on performance metrics. 
# 
# Here we can able to see that PPE is close to the left end of the box which means the data is left skewed.

# ### Scatter Plot for data point analysis

# In[80]:


plt.scatter(X, Y)
plt.xlabel('PPE')
plt.ylabel('motor_UPDRS')
plt.title('PPE vs motor_UPDRS')
plt.show()


# We can clearly see that the data points are widespread and also the presence of outliers which indicates that the relationship between the two variables is complex and may not be well-described by a simple linear model.
# 
# The scatter of data points across a broad range of values indicates that there may not be a strong or clear linear relationship between the two variables

# ### Load data into input and target variable into an array of shape nX1

# In[81]:


X1 = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1,1)


# ### Train, Test and Prediction  Metrics

# In[82]:


model = LR_1var(data,X1,Y,0.7,0.004,100000,0.000001)
model.train()
model.predict()


# ### Observation

# From the above metrics we can observe that R squared error is negative which means our model is a poor fit.

# In[83]:


model.plot_cost_history()
model.plot_regression()


# ### Cost function observation

# When the cost function is going down during training, it means the model is getting better. It's like the model is trying to make fewer mistakes or reach its goal. But, just because the cost is decreasing, it doesn't mean the model is perfect. It could still have some room to improve. So, it's important to keep an eye on the cost to make sure the model is learning well and getting closer to what we want it to do.

# In[ ]:




