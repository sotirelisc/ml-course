# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset
dataset = pd.read_csv('Position_Salaries.csv')

# Prepare data
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Not enough data, so no need to split dataset

# Fit LR to dataset
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression()
linear_regressor.fit(X, y)

# Fit PR to dataset
from sklearn.preprocessing import PolynomialFeatures
# Degree 4 makes better predections
poly_regressor = PolynomialFeatures(degree = 4)
# Turn X to X_poly with columns of powers
X_poly = poly_regressor.fit_transform(X)

# Fit 2nd LR to X_poly, y
linear_regressor_2 = LinearRegression()
linear_regressor_2.fit(X_poly, y)

# Visualize LR results

# Will always be a straight line, bad prediction
plt.scatter(X, y, color = 'red')
plt.plot(X, linear_regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualize PR results

# More detailed view of curve / smoother
X_grid = np.arange(min(X), max(X), 0.1)
# We need a matrix not a vector
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linear_regressor_2.predict(poly_regressor.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Predict a new result with LR
linear_regressor.predict(6.5)

# Predict a new result with PR
linear_regressor_2.predict(poly_regressor.fit_transform(6.5))