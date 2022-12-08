# Polynomial regression showing the relationship between milkshake sizes and prices

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Training data
x_train =  [[3], [5], [7], [11], [15]]     # Size of milkshake cups
y_train = [[4], [6], [10], [14], [15]]       # Prices of milkshakes

# Testing data
x_test = [[3], [5], [8], [13]]       # Milileter of milkshakes
y_test = [[5], [9], [12], [15]]       # Prices of milkshakes

# Training the Linear Regression model as well as plotting a prediction
regression = LinearRegression()
regression.fit(x_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regression.predict(xx.reshape(xx.shape[0], 1))
plt.plot(xx, yy)

# Setting the degree of the Polynomial Regression model
quadratic = PolynomialFeatures(degree = 2)

# Transforming input data matrix into a new data matrix of a given degree using this preprocessor
x_train_quad = quadratic.fit_transform(x_train)
x_test_quad = quadratic.transform(x_test)

# Training and testing the regression model
regression_quad = LinearRegression()
regression_quad.fit(x_train_quad, y_train)
xx_quad = quadratic.transform(xx.reshape(xx.shape[0], 1))

# Plotting the graph
plt.plot(xx, regression_quad.predict(xx_quad), c = 'r', linestyle = '--')
plt.title("Milkshake price regression on diameter")
plt.xlabel("Contents Milileter")
plt.ylabel("Price in Rand")
plt.axis([0, 25, 0, 25])
plt.grid(True)
plt.scatter(x_train, y_train)
plt.show()
