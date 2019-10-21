import numpy as np
#Gradient Descent Formula: theta = theta - (1/samples) * learning_rate*((Z-y)*X[i])
def gradient_Descent(X,y,theta,learning_rate = 0.05,max_iterations = 10000):
    m = len(y)
    cost_history = np.zeros(max_iterations)
    theta_history = np.zeros((max_iterations,3))
    for i in range(max_iterations):
        #1/m * X.T (z - y)
        z = sigmoid(np.dot(X,theta))
        theta = theta - (1/m)*learning_rate*(X.T.dot(z-y))
        theta_history[i,:] = theta.T
        cost_history[i] = cost_function(X,y,theta)
    return theta,cost_history,theta_history
def cost_function(X,y,theta):
    m = len(y)
    prediction = sigmoid(np.dot(X,theta))
    cost = (1/m)*np.sum((-y.T*np.log(prediction)-(1-y).T*np.log(1-prediction)))
    return cost
def sigmoid(z):
    return 1/(1+(np.e**(-z)))