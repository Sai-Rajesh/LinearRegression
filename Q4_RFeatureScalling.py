#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


data = pd.read_csv("parkinsons_dataset.csv")
data = data.dropna()


# In[2]:


from LinearRegression_Regularized import LR_4var

lamda_tp = [10, 1, 0.1, 0.01,0.001,0.0001,0.00001]
mse_l = []
r2_err_l = []
adj_r2_err_l = []

for i in lamda_tp:
    X1 = np.array(data['age'].values).reshape(-1,1)
    X2 = np.array(data['HNR'].values).reshape(-1,1)
    X3 = np.array(data['RPDE'].values).reshape(-1,1)
    X4 = np.array(data['DFA'].values).reshape(-1,1)
    Y = np.array(data['PPE'].values).reshape(-1,1)
    
    split_ratio = 0.7       # Train-test split ratio
    learning_rate = 0.000005     # Learning rate
    num_iter = 50000       # Number of iterations
    threshold = 0.000001 
    model = LR_4var(data,X1,X2,X3,X4,Y,split_ratio,learning_rate,i,num_iter,threshold)
    model.train()
    mse,r2_err,adj_r2_err = model.predict()
    mse_l.append(mse)
    r2_err_l.append(r2_err)
    adj_r2_err_l.append(adj_r2_err)

pd_data = {
    'MSE': mse_l,
    'R2 Error': r2_err_l,
    'Adj R2 Error': adj_r2_err_l,
    'Lambda': lamda_tp
}

metrics = pd.DataFrame(pd_data)


# ### Obseravtion
# 
# when regularization does not lead to a significant improvement in model metrics or performance. The regularization process involved training the model with various lambda values and evaluating their performance using metrics such as Mean Squared Error (MSE), R-squared Error (R2 Error), and Adjusted R-squared Error (Adj R2 Error). The lambda values tested were [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001].
# 
# regularization improves the performance if the chosen lambda value leads to lower MSE and higher R2 and Adj R2 values compared to the non-regularized model. The metrics for each lambda in the metrics DataFrame to identify the lambda value associated with the best overall performance.
# 
# The consistency in convergence and the stability of MSE, R-squared, and Adjusted R-squared across different lambda values suggest that regularization does not significantly impact the model's performance, The negative values of R-squared indicate that the model might not be well-suited for predicting the dependent variable.
# 
# The regularization process did not lead to substantial changes in model performance. The model's fit to the data remains limited, as indicated by the negative R-squared values.

# ### Q4 B - Feature Scaling

# In[3]:


# Assuming this is the start of your code
from LinearRegression import LR_4var
import numpy as np

X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['HNR'].values).reshape(-1, 1)
X3 = np.array(data['RPDE'].values).reshape(-1, 1)
X4 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['PPE'].values).reshape(-1, 1)

# Feature Scaling
X1 = (X1 - np.mean(X1)) / (np.max(X1) - np.min(X1))
X2 = (X2 - np.mean(X2)) / (np.max(X2) - np.min(X2))
X3 = (X3 - np.mean(X3)) / (np.max(X3) - np.min(X3))
X4 = (X4 - np.mean(X4)) / (np.max(X4) - np.min(X4))

split_ratio = 0.7
learning_rate = 0.0001
num_iter = 10000000
threshold = 0.000001
model = LR_4var(data, X1, X2, X3, X4, Y, split_ratio, learning_rate, num_iter, threshold)
model.train()
mse, r2_err, adj_r2_err = model.predict()


# ### Observation
# Feature scaling to standardize the range of independent features. It can improve the performance of some models by ensuring that all features contribute equally to the learning process.
# 
# The MSE measures the average squared difference between predicted and actual values and after feature scaling, the MSE is approximately 0.0302.
# 
# The R-squared Error indicates the proportion of the variance in the independent variables. The R-squared Error value is negative, which suggests that the model may fit the data worse than a horizontal line. Specifically, the R-squared Error is approximately -1.761.
# 
# Adjusted R-squared Error adjusts the R-squared value based on the number of predictors in the model. Similar to the R-squared Error, the Adjusted R-squared Error is negative, depicts a poor model fit.The Adjusted R-squared Error is approximately -1.767.
# 
# Feature scaling did not lead to a huge improvement in model performance, as indicated by the MSE, R-squared Error, and Adjusted R-squared Error. The negative values of R-squared and Adjusted R-squared suggest that the model might not be well-suited for predicting the dependent variable based on the given features, regardless of feature scaling.

# In[ ]:




