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