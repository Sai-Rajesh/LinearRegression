#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('parkinsons_dataset.csv')


# In[3]:


import numpy as np

class StepwiseRegression:
    def __init__(self, X, Y, features, split_ratio=0.8, learning_rate=0.000001, num_iterations=1000, 
                 convergence_threshold=0.000001):
        self.X = X
        self.Y = Y
        self.features = features
        self.split_ratio = split_ratio
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.convergence_threshold = convergence_threshold
        self.theta = np.zeros((X.shape[1], 1))

    def split_data(self):
        split_index = int(len(self.X) * self.split_ratio)
        X_train, X_test = self.X[:split_index], self.X[split_index:]
        Y_train, Y_test = self.Y[:split_index], self.Y[split_index:]
        return X_train, Y_train, X_test, Y_test

    def v_cost_function(self, predictions, Y):
        cost = (1 / (2 * len(Y))) * np.sum((predictions - Y) ** 2)
        return cost

    def train(self):
        X_train, Y_train, _, _ = self.split_data()

        cost_history = []

        for i in range(self.num_iterations):
            predictions = np.dot(X_train, self.theta)
            cost = self.v_cost_function(predictions, Y_train)
            cost_history.append(cost)

            gradients = (1 / len(Y_train)) * np.dot(X_train.T, predictions - Y_train)

            self.theta -= self.learning_rate * gradients

            # Check convergence
            if i > 0 and abs(cost_history[i] - cost_history[i - 1]) < self.convergence_threshold:
                print(f"Converged after {i + 1} iterations.")
                break

        return cost_history

    def predict_and_evaluate(self):
        _, _, X_test, Y_test = self.split_data()

        # Calculate predicted values using the linear regression model
        predictions_test = np.dot(X_test, self.theta)

        print("Features:", ', '.join(f"({i + 1}) {feature}" for i, feature in enumerate(self.features)))

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((predictions_test - Y_test) ** 2)
        print("Mean Squared Error (MSE):", mse)

        # Calculate R-squared Error
        ssr = np.sum((predictions_test - Y_test) ** 2)
        sst = np.sum((Y_test - np.mean(Y_test)) ** 2)

        # Check if sst is zero to avoid division by zero
        if sst == 0:
            print("The value of sst is 0, indicating that the independent variable/s is not suitable for prediction!")
        else:
            r_sqErr = 1 - (ssr / sst)
            print("R-squared Error:", r_sqErr)

        # Calculate Adjusted R-squared Error
        m = len(Y_test)
        num_features = self.X.shape[1]
        adj_r_sqErr = 1 - (1 - r_sqErr) * (m - 1) / (m - num_features - 1)
        print("Adjusted R-squared Error:", adj_r_sqErr)
        print("\n")


# StepWise Backward Regression:
# The StepwiseRegression class provides a structured approach to perform stepwise regression, allowing for model training, evaluation, and interpretation. It facilitates the exploration of feature subsets and their impact on predictive performance. 
# 
# Model with a negative R-squared Error and Adjusted R-squared Error, indicating poor fit to the data.

# With 10 Features

# In[4]:


feature = ['age', 'Jitter(%)', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ5',
'Shimmer:APQ11', 'HNR', 'RPDE', 'DFA', 'PPE']

# Create feature matrix and target variable
X = np.array([data[f].values.reshape(-1, 1) for f in feature])
X = np.concatenate(X, axis=1)
Y = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

dynamic_stepwise_regression = StepwiseRegression(X, Y, feature)
cost_history = dynamic_stepwise_regression.train()
dynamic_stepwise_regression.predict_and_evaluate()


# The high MSE depicts us that it is a substantial spread between predicted and actual values and suggests inadequacy of the model.

# Here we have implemented above regression using 9 feautures

# In[5]:


# List of feature names
features = ['age', 'Jitter(%)', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ5', 'Shimmer:APQ11', 'HNR', 'RPDE', 'DFA', 'PPE']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# The model seems to exhibit poor fit, indicated by negative R-squared values and high MSE.

# Here we have implemented above regression using 8 feautures

# In[6]:


# List of feature names
features = ['age', 'Jitter(%)', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ5', 'HNR', 'RPDE', 'DFA', 'PPE']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Excluding 'PPE' maintains stability in metrics, depicts that it has limited impact on the overall model.

# Here we have implemented above regression using 7 feautures

# In[7]:


# List of feature names
features = ['age', 'Shimmer', 'Shimmer(dB)', 'Shimmer:APQ5', 'HNR', 'RPDE', 'DFA', 'PPE']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Features related to shimmer ('Shimmer,' 'Shimmer(dB),' 'Shimmer:APQ5') gives average model's predictive efficiency.
# 'Age' plays a crucial role in improving the model's performance, as indicated by the substantial increase in MSE and decrease in R-squared metrics when excluded and other features ('HNR,' 'RPDE,' 'DFA,' 'PPE') have less impact on the model's performance.

# Here we have implemented above regression using 6 feautures

# In[8]:


# List of feature names
features = ['age', 'Shimmer', 'Shimmer(dB)', 'HNR', 'RPDE', 'DFA', 'PPE']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Negative R-squared values across iterations indicate that the selected features, regardless of exclusions, depicts the variability in the target variable.

# Here we have implemented above regression using 5 feautures

# In[9]:


# List of feature names
features = ['age', 'Shimmer(dB)', 'HNR', 'RPDE', 'DFA', 'PPE']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Excluding 'Shimmer(dB)' appears to have a positive impact on model performance, stating the importance of this feature and
# other feature exclusions show varying effects, with 'age' having a significant negative impact on the model.

# Here we have implemented above regression using 4 feautures

# In[10]:


# List of feature names
features = ['age', 'Shimmer(dB)', 'HNR', 'RPDE', 'DFA']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Excluding 'age' significantly impairs the model, while the exclusion of other features has a relatively minor impact and in the case of 'Shimmer(dB),' leads to slight improvement. The negative R-squared values highlight challenges in explaining the variability in the target variable.

# Here we have implemented above regression using 3 feautures

# In[11]:


# List of feature names
features = ['age', 'Shimmer(dB)', 'HNR', 'DFA']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# The inclusion of 'Shimmer(dB)' appears to improve model performance, as seen in lower MSE and less negative R-squared values and the exclusion of 'age' significantly deteriorates the model, emphasizing its importance.'HNR' and 'DFA' removals show minor impacts on model performance.

# Here we have implemented above regression using 2 feautures

# In[12]:


# List of feature names
features = ['age', 'Shimmer(dB)', 'HNR']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# Here,'age' appears to be a crucial feature, as its exclusion leads to a substantial deterioration in model performance and excluding 'Shimmer(dB)' results in a huge improvement in model performance.'HNR' and 'Shimmer(dB)' show comparable impacts when excluded, with slightimprovement in model performance.

# Here we have implemented above regression using 1 feautures

# In[13]:


# List of feature names
features = ['age', 'HNR']

for excluded_feature in features:
    print("Excluded feature:", excluded_feature)
    selected_features = [feature for feature in features if feature != excluded_feature]
    
    # Create a copy of the original feature matrix for each iteration
    X_temp = np.array([data[feature].values.reshape(-1, 1) for feature in selected_features])
    X_temp = np.concatenate(X_temp, axis=1)
    Y_temp = np.array(data['motor_UPDRS'].values).reshape(-1, 1)

    # Apply stepwise_regression to the copied feature matrix
    dynamic_stepwise_regression = StepwiseRegression(X_temp, Y_temp, selected_features)
    cost_history = dynamic_stepwise_regression.train()
    dynamic_stepwise_regression.predict_and_evaluate()


# We have implemented stepwise regression by excluding each feature, one at a time, from the original feature set of 'age' and 'HNR'. The output includes the excluded feature, cost history, and evaluation metrics (Mean Squared Error, R-squared Error, Adjusted R-squared Error) for each iteration. 

# When 'age' is excluded,with single feature 'HNR,' the model's effectiveness gets significantly deteriorates, which depicts high Mean Squared Error (MSE) and extremely negative R-squared values, indicating a poor fit to the data and inversely excluding 'HNR' with single feature 'age' results in a slight improvement, with a reduced MSE and less negative R-squared values. 

# Q3 C
# 
# The forward stepwise regression converged after 5875 and 3643 iterations for two different runs. The Mean Squared Error (MSE) is 90.4853, and the R-squared and Adjusted R-squared are around -0.70, indicating a moderate fit to the data. The selected features include 'age' and 'HNR', with corresponding coefficients (theta values).
# 
# Two iterations are shown in Backward Regression, each excluding one feature ('age' or 'HNR') at a time. The MSE is higher compared to forward regression, with values of 462.7413 and 111.8758, indicating increased error. The remaining feature after exclusion is 'HNR' or 'age' in each iteration, with corresponding coefficients.
# 
# Forward regression took more iterations to converge, suggesting that it required additional steps to refine the model.Forward regression resulted in a model with lower MSE and relatively better R-squared and Adjusted R-squared values.
# Backward regression yielded higher MSE, suggesting a less accurate model fit. Backward regression sequentially excluded 'age' and 'HNR', potentially leading to a simpler model but at the cost of increased error. 
# 
# Forward regression prioritizes inclusion, potentially capturing more nuanced relationships, while backward regression seeks simplicity by iteratively excluding features.

# Q3 D
# 
# Mean Squared Error values for both MSE values approximately equal to 90.69. The similarity in MSE suggests that the models, irrespective of the feature set, are making predictions with similar levels of error.
# 
# The R-squared Error measures the proportion of the variance in the dependent variable that is predictable from the independent variables and R-squared Error is approximately -0.70. This value implies that the model does not explain much of the variance in the motor_UPDRS variable. The similarity in R-squared Error indicates that the choice of features does not significantly impact the model's ability to explain the variability in motor_UPDRS.
# 
# The Adjusted R-squared Error adjusts the R-squared value based on the number of predictors in the model these models have an Adjusted R-squared Error close to -0.70, indicating that the adjustment for the number of predictors does not lead to a substantial difference.
# 
# As per my observation, MSE, R-squared Error, and Adjusted R-squared Error suggests that the performance of the two models is similar. The selected features from backward stepwise regression do not appear to provide advantage over the initially provided features in predicting the motor_UPDRS variable. The choice of features does not significantly impact the predictive accuracy of the linear regression model for motor_UPDRS.

# In[ ]:





# In[ ]:




