#!/usr/bin/env python
# coding: utf-8

# In[28]:


from LinearRegression import LR_1var
import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sn


data = pd.read_csv("parkinsons_dataset.csv")
data = data.dropna()


# for this forward regression i have choosed age, hnr, rpde, dfa, ppe

# ### Linear Regression with one var

# In[29]:


# Initialize lists to store metrics and model parameters (1-variable case)
metrics = []              # List for metrics
mse_1var = []             # List for Mean Squared Error
r2_err_1var = []          # List for R-squared Error
adj_r2_err_1var = []      # List for Adjusted R-squared Error
col_name = []             # List to store column names
learning_rates = []       # List to store learning rates

# Import the Linear Regression model from the 'LinearRegression' module
from LinearRegression import LR_1var

# Prepare the independent and dependent variables from the dataset (1-variable case 1)
X1 = np.array(data['age'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (1-variable case 1)
split_ratio = 0.7       # Train-test split ratio
learning_rate = 0.000005     # Learning rate
num_iter = 50000       # Number of iterations
threshold = 0.000001    # Convergence threshold

# Initialize the Linear Regression model with the data and hyperparameters (1-variable case 1)
model = LR_1var(data, X1, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (1-variable case 1)
model.train()

# Predict and calculate metrics (1-variable case 1)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()
# Store the column names and metrics (1-variable case 1)
col_name.append('age')
mse_1var.append(mse)
r2_err_1var.append(r2_err)
adj_r2_err_1var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (1-variable case 2)
X1 = np.array(data['RPDE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (1-variable case 2)
split_ratio = 0.7       
learning_rate = 0.05   
num_iter = 100000      
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (1-variable case 2)
model = LR_1var(data, X1, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (1-variable case 2)
model.train()

# Predict and calculate metrics (1-variable case 2)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (1-variable case 2)
col_name.append('RPDE')
mse_1var.append(mse)
r2_err_1var.append(r2_err)
adj_r2_err_1var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (1-variable case 3)
X1 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (1-variable case 3)
split_ratio = 0.7       
learning_rate = 0.0002  
num_iter = 100000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (1-variable case 3)
model = LR_1var(data, X1, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (1-variable case 3)
model.train()

# Predict and calculate metrics (1-variable case 3)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()
# Store the column names and metrics (1-variable case 3)
col_name.append('DFA')
mse_1var.append(mse)
r2_err_1var.append(r2_err)
adj_r2_err_1var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (1-variable case 4)
X1 = np.array(data['PPE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (1-variable case 4)
split_ratio = 0.7       
learning_rate = 0.0002  
num_iter = 100000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (1-variable case 4)
model = LR_1var(data, X1, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (1-variable case 4)
model.train()

# Predict and calculate metrics (1-variable case 4)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (1-variable case 4)
col_name.append('PPE')
mse_1var.append(mse)
r2_err_1var.append(r2_err)
adj_r2_err_1var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (1-variable case 5)
X1 = np.array(data['Shimmer'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (1-variable case 5)
split_ratio = 0.7       
learning_rate = 0.0002  
num_iter = 100000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (1-variable case 5)
model = LR_1var(data, X1, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (1-variable case 5)
model.train()

# Predict and calculate metrics (1-variable case 5)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()
# Store the column names and metrics (1-variable case 5)
col_name.append('Symmetry')
mse_1var.append(mse)
r2_err_1var.append(r2_err)
adj_r2_err_1var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Create a DataFrame to display the metrics (1-variable case)
metrics_1var = pd.DataFrame({
    "ivar": col_name, "lr": learning_rates, "mse": mse_1var, "r2_err": r2_err_1var, "adj_r2_err": adj_r2_err_1var
})

# Display the first few rows of the metrics DataFrame (1-variable case)
metrics_1var.head()


# RPDE stands out as a potentially crucial feature based on its higher mse and adjusted r squared error.

# ### Linear Regression with 2 vars

# In[30]:


# Initialize lists to store metrics and model parameters (2-variable case)
mse_2var = []           
r2_err_2var = []        
adj_r2_err_2var = []    
col_name = []           
learning_rates = []     

# Import the Linear Regression model from the 'LinearRegression' module
from LinearRegression import LR_2var

# Prepare the independent and dependent variables from the dataset (2-variable case 1)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['RPDE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (2-variable case 1)
split_ratio = 0.7       
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001   

# Initialize the Linear Regression model with the data and hyperparameters (2-variable case 1)
model = LR_2var(data, X1, X2, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (2-variable case 1)
model.train()

# Predict and calculate metrics (2-variable case 1)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (2-variable case 1)
col_name.append('[age, RPDE]')
mse_2var.append(mse)
r2_err_2var.append(r2_err)
adj_r2_err_2var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (2-variable case 2)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (2-variable case 2)
split_ratio = 0.7       
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (2-variable case 2)
model = LR_2var(data, X1, X2, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (2-variable case 2)
model.train()

# Predict and calculate metrics (2-variable case 2)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (2-variable case 2)
col_name.append('[age, DFA]')
mse_2var.append(mse)
r2_err_2var.append(r2_err)
adj_r2_err_2var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (2-variable case 3)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['PPE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (2-variable case 3)
split_ratio = 0.7      
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (2-variable case 3)
model = LR_2var(data, X1, X2, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (2-variable case 3)
model.train()

# Predict and calculate metrics (2-variable case 3)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (2-variable case 3)
col_name.append('[age, PPE]')
mse_2var.append(mse)
r2_err_2var.append(r2_err)
adj_r2_err_2var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (2-variable case 4)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (2-variable case 4)
split_ratio = 0.7    
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (2-variable case 4)
model = LR_2var(data, X1, X2, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (2-variable case 4)
model.train()

# Predict and calculate metrics (2-variable case 4)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (2-variable case 4)
col_name.append('[age, Shimmer]')
mse_2var.append(mse)
r2_err_2var.append(r2_err)
adj_r2_err_2var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Create a DataFrame to display the metrics
metrics_2var = pd.DataFrame({
    "ivar": col_name, "lr": learning_rates, "mse": mse_2var, "r2_err": r2_err_2var, "adj_r2_err": adj_r2_err_2var
})

# Display the first few rows of the metrics DataFrame
metrics_2var.head()


# Age and Shimmer appears to perform better than other features, as it has the lowest MSE and relatively higher R-squared and Adjusted R-squared values.

# ### Linear Regression with three Var

# In[31]:


# Import the Linear Regression model from the 'LinearRegression' module
from LinearRegression import LR_3var

# Initialize lists to store metrics and model parameters
mse_3var = []          
r2_err_3var = []        
adj_r2_err_3var = []    
col_name = []           
learning_rates = []     

# Prepares the independent and dependent variables from the dataset (3-variable case 1)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['RPDE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (3-variable case 1)
split_ratio = 0.7       
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001   

# Initialize the Linear Regression model with the data and hyperparameters (3-variable case 1)
model = LR_3var(data, X1, X2, X3, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (3-variable case 1)
model.train()

# Predict and calculate metrics (3-variable case 1)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (3-variable case 1)
col_name.append(str('[age, Shimmer, RPDE]'))
mse_3var.append(mse)
r2_err_3var.append(r2_err)
adj_r2_err_3var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepare the independent and dependent variables from the dataset (3-variable case 2)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (3-variable case 2)
split_ratio = 0.7     
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001   

# Initialize the Linear Regression model with the data and hyperparameters (3-variable case 2)
model = LR_3var(data, X1, X2, X3, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (3-variable case 2)
model.train()

# Predict and calculate metrics (3-variable case 2)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (3-variable case 2)
col_name.append(str('[age, Shimmer, DFA]'))
mse_3var.append(mse)
r2_err_3var.append(r2_err)
adj_r2_err_3var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Prepares the independent and dependent variables from the dataset (3-variable case 3)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['PPE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (3-variable case 3)
split_ratio = 0.7     
learning_rate = 0.000005     
num_iter = 50000       
threshold = 0.000001  

# Initialize the Linear Regression model with the data and hyperparameters (3-variable case 3)
model = LR_3var(data, X1, X2, X3, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (3-variable case 3)
model.train()

# Predict and calculate metrics (3-variable case 3)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (3-variable case 3)
col_name.append(str('[age, Shimmer, PPE]'))
mse_3var.append(mse)
r2_err_3var.append(r2_err)
adj_r2_err_3var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Create a DataFrame to display the metrics
metrics_3var = pd.DataFrame({
    "ivar": col_name, "lr": learning_rates, "mse": mse_3var, "r2_err": r2_err_3var, "adj_r2_err": adj_r2_err_3var
})

# Display the first few rows of the metrics DataFrame
metrics_3var.head()


# age, Shimmer and RPDE has performed well then the other models.

# ### Linear Regression with 4 variables

# In[32]:


# Import the Linear Regression model from the 'LinearRegression' module
from LinearRegression import LR_4var

# Initialize lists to store metrics and model parameters
mse_4var = []        
r2_err_4var = []       
adj_r2_err_4var = []  
col_name = []           
learning_rates = []    

# Prepare the independent and dependent variables from the dataset (4-variable case 1)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['RPDE'].values).reshape(-1, 1)
X4 = np.array(data['PPE'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (4-variable case 1)
split_ratio = 0.7      
learning_rate = 0.000005    
num_iter = 50000      
threshold = 0.000001   

# Initialize the Linear Regression model with the data and hyperparameters (4-variable case 1)
model = LR_4var(data, X1, X2, X3, X4, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (4-variable case 1)
model.train()

# Predict and calculate metrics (4-variable case 1)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics (4-variable case 1)
col_name.append(str('[age, Shimmer, RPDE, PPE]'))
mse_4var.append(mse)
r2_err_4var.append(r2_err)
adj_r2_err_4var.append(adj_r2_err)
learning_rates.append(learning_rate)


# Prepare the independent and dependent variables from the dataset (4-variable case 2)
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['RPDE'].values).reshape(-1, 1)
X4 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters (4-variable case 2)
split_ratio = 0.7       
learning_rate = 0.000005     
num_iter = 50000      
threshold = 0.000001    

# Initialize the Linear Regression model with the data and hyperparameters (4-variable case 2)
model = LR_4var(data, X1, X2, X3, X4, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model (4-variable case 2)
model.train()

# Predict and calculate metrics (4-variable case 2)
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()
# Store the column names and metrics (4-variable case 2)
col_name.append(str('[age, Shimmer, RPDE, DFA]'))
mse_4var.append(mse)
r2_err_4var.append(r2_err)
adj_r2_err_4var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Create a DataFrame to display the metrics
metrics_4var = pd.DataFrame({
    "ivar": col_name, "lr": learning_rates, "mse": mse_4var, "r2_err": r2_err_4var, "adj_r2_err": adj_r2_err_4var
})

# Display the first few rows of the metrics DataFrame
metrics_4var.head()


# age, shimmer, RPDE, DFA has performed well then the other models.

# ### Linear Regression with 5vars

# In[33]:


# Import the Linear Regression model from the 'LinearRegression' module
from LinearRegression import LR_5var

# Initialize lists to store metrics and model parameters
mse_5var = []         
r2_err_5var = []       
adj_r2_err_5var = []    
col_name = []          
learning_rates = []     

# Prepare the independent and dependent variables from the dataset
X1 = np.array(data['age'].values).reshape(-1, 1)
X2 = np.array(data['Shimmer'].values).reshape(-1, 1)
X3 = np.array(data['RPDE'].values).reshape(-1, 1)
X4 = np.array(data['PPE'].values).reshape(-1, 1)
X5 = np.array(data['DFA'].values).reshape(-1, 1)
Y = np.array(data['HNR'].values).reshape(-1, 1)

# Set model hyperparameters
split_ratio = 0.7     
learning_rate = 0.000005    
num_iter = 50000      
threshold = 0.000001   

# Initialize the Linear Regression model with the data and hyperparameters
model = LR_5var(data, X1, X2, X3, X4, X5, Y, split_ratio, learning_rate, num_iter, threshold)

# Train the model
model.train()

# Predict and calculate metrics
mse, r2_err, adj_r2_err = model.predict()
model.plot_cost_history()

# Store the column names and metrics
col_name.append(str('[age, Shimmer, RPDE, PPE, DFA]'))
mse_5var.append(mse)
r2_err_5var.append(r2_err)
adj_r2_err_5var.append(adj_r2_err)
learning_rates.append(learning_rate)

# Create a DataFrame to display the metrics
metrics_5var = pd.DataFrame({
    "ivar": col_name, "lr": learning_rates, "mse": mse_5var, "r2_err": r2_err_5var, "adj_r2_err": adj_r2_err_5var
})

# Display the first few rows of the metrics DataFrame
metrics_5var.head()


# ### Observation

# As per my observation, performance cannot be evaluated specifically based on single model and it takes few more features for better performance

# In[ ]:




