# Making the imports
import numpy as np
from numpy.linalg import inv
import pandas as pd
import matplotlib.pyplot as plt
import time

plt.rcParams['figure.figsize'] = (8.0, 6.0)

# Preprocessing Input data
data = pd.read_csv('data1.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.show()
plt.scatter(X, Y)

def gradient_descent():
    # Building the model
    m = 0
    c = 0
    L = 0.0001  # The learning Rate
    epochs = 1000  # The number of iterations to perform gradient descent
    n = float(len(X)) # Number of elements in X
    # Performing Gradient Descent
    tic = time.perf_counter()
    for i in range(epochs):
        Y_pred = m*X + c  # The current predicted value of Y
        D_m = (-2/n) * sum(X * (Y - Y_pred))  # Derivative wrt m
        D_c = (-2/n) * sum(Y - Y_pred)  # Derivative wrt c
        m = m - L * D_m  # Update m
        c = c - L * D_c  # Update c
    runtime = time.perf_counter() - tic
    return [m,c, runtime]



def matrix_method():
    data_matrix = data.to_numpy()
    Y_matrix = np.zeros((len(data_matrix),1), dtype = float)
    for i in range(len(data_matrix)):
        Y_matrix[i][0] = data_matrix[i][1]
        data_matrix[i][1] = 1
    tic = time.perf_counter()
    coefficent_matrix = np.dot(inv(np.dot(data_matrix.T, data_matrix)),np.dot(data_matrix.T, Y_matrix))
    runtime = time.perf_counter() - tic
    final_matrix = np.append(coefficent_matrix,runtime)
    return final_matrix

answer1 = gradient_descent()
answer2 = matrix_method()

Y_pred = answer1[0]*X + answer1[1]
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted

Y_pred = answer2[0]*X + answer2[1]
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='green') # predicted
print("Runtime for gradient ", answer1[2])
print("Runtime for matrix ", answer2[2])
plt.show()
