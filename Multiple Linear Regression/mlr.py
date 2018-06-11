# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('50_Startups.csv')

# Prepare data
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode categorical data

# Encode IV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
# Always fit & transform encoder to our dataset's column
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

# Transform 'State' column to 3 different columns with 0 or 1 values (true or false)
# Use index of 'State' column (3)
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid the Dummy Variable Trap

# Keep all lines but remove 1st column (REMEMBER: Keep n-1 dummy vars)
X = X[:, 1:]

# Training set & Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fit MLR to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_prediction = regressor.predict(X_test)

# Build the optimal model using Backward Elimination
import statsmodels.formula.api as sm

# Add b0 constant (required by statsmodels lib, others handle it)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x

# Set Significance Level
SL = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
X_modeled = backwardElimination(X_optimal, SL)

# BACKWARD ELIMINATION ALGORITHM STEP BY STEP

X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
# Create new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
# Find P-values and choose var with the max one
regressor_OLS.summary()

# Eliminate it
X_optimal = X[:, [0, 1, 3, 4, 5]]
# Create new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
# Find P-values and choose var with the max one
regressor_OLS.summary()

# Eliminate it
X_optimal = X[:, [0, 3, 4, 5]]
# Create new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
# Find P-values and choose var with the max one
regressor_OLS.summary()

# Eliminate it
X_optimal = X[:, [0, 3, 5]]
# Create new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
# Find P-values and choose var with the max one
regressor_OLS.summary()

# Eliminate it
X_optimal = X[:, [0, 3]]
# Create new regressor
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
# Find P-values and choose var with the max one
regressor_OLS.summary()
