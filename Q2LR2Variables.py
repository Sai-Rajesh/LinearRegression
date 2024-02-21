#!/usr/bin/env python
# coding: utf-8

# In[84]:


# import the required libraries
from LinearRegression import LR_2var
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sn


# ### Load the data

# In[85]:


# read the data from csv using pandas
data = pd.read_csv("parkinsons_dataset.csv")


# 
# ### Analyze the data

# In[86]:


data[['PPE','NHR','motor_UPDRS']].describe()


# This extracts data of two columns features i:e; PPE and motor_UPDRS from data frame.

# ### Drop null rows

# In[87]:


data = data.dropna(axis=0)


# This helps to remove null values if exists

# ### Data Visualization
# 

# In[88]:


X1 = data['PPE'].values
X2 = data['NHR'].values
Y = data['motor_UPDRS'].values
plt.figure(figsize=(20, 6)) 

plt.subplot(3,1, 1)
plt.boxplot(X1, vert=False)
plt.title('PPE')

plt.subplot(3, 1, 2)
plt.boxplot(X2, vert=False)  
plt.title('NHR')

plt.subplot(3, 1, 3)
plt.boxplot(Y, vert=False)  
plt.title('motor_UPDRS')

plt.show()


# In "NHR" we can see the data is mostly left skewed and there are many outliers in the data points which will show great impact on data analysis. This depicts us that the minimum value is very close to the first quartile (Q1) and this will have have negative impact on performance metrics. 

# ### Scatter Plot

# In[89]:


# Sample data (replace these with your data)
X1 =  np.array(data['PPE'].values).reshape(-1,1)
X2 =  np.array(data['NHR'].values).reshape(-1,1)
Y = np.array(data['motor_UPDRS'].values).reshape(-1,1)

# Create a scatter plot
plt.figure(figsize=(6, 6))  # Adjust figure size if needed
plt.scatter(X1, Y, label='PPE vs NHR', marker='o', c='blue', alpha=0.7)
plt.scatter(X2, Y, label='motor_UPDRS vs NHR', marker='x', c='red', alpha=0.7)

# Customize plot
plt.title('Scatter Plot')
plt.xlabel('[PPE,NHR ]')
plt.ylabel('motor_UPDRS')
plt.legend()

# Show the plot
plt.grid(True)
plt.show()


# This range of Spread in data points of outliers suggest that the association between the two variables is intricate and might not be effectively characterized by a single linear model. The scattering of data points across a wide range implies that there may not be  distinct linear connection among the three variables.

# ### Load Data into input and target variables in the format of an array of shape nX1

# In[90]:


X1 = np.array(data['PPE'].values)
X2 = np.array(data['NHR'].values)
Y = np.array(data['motor_UPDRS'].values)


# In[91]:


# model_2var.X2_test
#
#
# # In[92]:
#
#
# model_2var.Y_test


# In[93]:


from LinearRegression import LR_2var

model = LR_2var(data,X1,X2,Y,0.7,0.004,100000,0.000001)
model.train()
model.predict()
model.plot_cost_history()
model.plot_regression()


# In[94]:


# Assuming LR_1var and LR_2var classes getting metrics, mean squared error

from LinearRegression import LR_1var
import matplotlib.pyplot as plt
import numpy as np

# Create instances of LR_2var and LR_1var
model_2var = LR_2var(data, X1, X2, Y, 0.7, 0.004, 100000, 0.000001)
model_1var = LR_1var(data,X1,Y,0.7,0.004,100000,0.000001)

# Train the models
model_2var.train()
model_1var.train()

# Make predictions
predictions_2var = model_2var.predict()
predictions_1var = model_1var.predict()

# Compare metrics
mse_2var = model_2var.predict()
mse_1var = model_1var.predict()

# Plot cost history for both models

model_2var.plot_cost_history()

model_1var.plot_cost_history()

predictions_2var= [model_2var.theta0 + model_2var.theta1 * i1 + model_2var.theta2 * i2 for i1, i2 in
                            zip(model_2var.X1_test, model_2var.X2_test)]
predictions_1var = [model_1var.theta0 + model_1var.theta1 * i for i in model_1var.X_test]

# Plot regression lines for both models
plt.figure(figsize=(10, 6))

plt.plot(model_2var.Y_test, predictions_2var, label='LR_2var Regression', c='red')
plt.plot(model_1var.Y_test, predictions_1var, label='LR_1var Regression', c='green')

plt.title('Linear Regression Comparison')
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.legend()
plt.show()

# Print and compare metrics
print(f"LR_2var Mean Squared Error: {mse_2var}")
print(f"LR_1var Mean Squared Error: {mse_1var}")


# Lower MSE values and well-fitted regression lines indicate better model performance.
# The visualizations and printed metrics collectively offer a comprehensive understanding of how effectively LR_2var and LR_1var capture the underlying patterns in the data.

# In[ ]:




