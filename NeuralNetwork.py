'''
Name: Timmy Lam ID: 214340814
Name: Jonathan Luk ID: 216640054
'''
#!/usr/bin/env python
# coding: utf-8

import numpy as np
import random as rm
#import seaborn as sns (Import this to the environment)
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix 
from sklearn.datasets import load_iris

def fit_NeuralNetwork(X_train, y_train, alpha, hidden_layer_sizes, epochs):
    weights = []
    err = []
    X = []
    S = []
    g = []
    nW = []
    totalerr = 0
    
    #Init weights
    weight = [[0.1 for x in range(len(X_train[0]) + 1)]for x in range(hidden_layer_sizes[0])]
    weights.append(weight)
        
    for i in range(len(hidden_layer_sizes) - 1):
        weight = [[0.1 for x in range(hidden_layer_sizes[i] + 1)]for x in range(hidden_layer_sizes[i + 1])]    
        weights.append(weight)
    
    weight = [0.1 for x in range(hidden_layer_sizes[i + 1] + 1)]
    weights.append(weight)

    #Apply the algorithms, update the weights, and compute the average error
    for i in range(epochs):
        #Randomzing at each epoch
        k = rm.randrange(len(X_train))
        X, S = forwardPropagation(X_train[k], weights)
        g = backPropagation(X, y_train[k], S, weights)
        weights = updateWeights(weights, g, alpha)
        totalerr = totalerr + errorPerSample(X, y_train[k])
        err.append(totalerr / (i + 1))

    return err, weights 

def errorPerSample(X, y_n):
    #Compute the error at the last layer
    return errorf(X[len(X) - 1], y_n)
    
def forwardPropagation(x, weights):
    X = []
    S = []
    #Account for the bias node
    x = np.hstack((1, x))
    
    for i in range(len(weights) - 1):
        X.append(x)
        #Calculate the new vector from the weight vector and x vector 
        z = np.dot((weights[i]), x)
        S.append(z)

        x = [0 for k in range(len(weights[i]))]
        
        #Run the vector through the activation function
        for j in range(len(weights[i])):
            x[j] = activation(z[j])
 
        x = np.hstack((1, x))

    X.append(x)
    z = np.dot((weights[i + 1]), x)
    S.append(z)
    #Run through the output function
    X.append(outputf(z))

    return X, S  

def backPropagation(X, y_n, s, weights):
    g = []
    #Computing the last layer
    delta = derivativeError(X[-1], y_n) * derivativeOutput(s[-1])
    g.insert(0,np.dot(delta,np.transpose(X[-2])))
    
    #Computing the backward messages 
    for i in range(2, len(weights) + 1):
        delta = np.dot(np.transpose(weights[-i + 1]), delta) 

        for j in range(len(delta)):
            delta[j] = delta[j] * np.hstack((1, derivativeActivationM(s[- i])))[j]
     
        if i == 2:
            delta = [delta]
        
        #Inserting the backward messages into a list after the final computation
        if np.shape(delta)[0] == np.shape(np.matrix(X[-i - 1]))[0]:
            delta = np.transpose(delta)
            delta = delta[1:]
            g.insert(0, np.dot(delta, [X[-i - 1]]))
        else:
            delta = delta[1:]
            g.insert(0, np.dot(delta, [X[-i - 1]]))
          
    return g

def updateWeights(weights, g, alpha):
    nW = weights
    #Use SGD to improve the weights
    for i in range(len(weights)):
        nW[i] = weights[i] - alpha * g[i]

    return nW

def activation(s):
    #ReLU outputs s if its greater than 0
    if s > 0:
        return s
    else:
        return 0
    
def derivativeActivation(s):
    #ReLU derivative outputs 1 if s is greater than 0
    if s > 0:
        return 1
    else:
        return 0
    
def derivativeActivationM(S):
    #ReLU derivative outputs 1 if s is greater than 0
    s=S
    for j in range(len(S)):
        s[j]=derivativeActivation(S[j])
            
    return s
    
def outputf(s):
    #Calculate the sigmoid function
    exp_s = 1 / (1 + np.exp(-s))
    return exp_s
    
def derivativeOutput(s):
    #Calculate the derivative of the sigmoid function
    exp_s = np.exp(-s) / (1 + np.exp(-s))**2
    return exp_s
    
def errorf(x_L, y):
    #Calculate the error function
    log_loss = 0
    
    if y == 1:
        log_loss = -np.log(x_L)
    elif y == -1:
        log_loss = -np.log(1 - x_L)
        
    return log_loss
    
def derivativeError(x_L, y):
    #Calculate the derivative of the error function
    log_loss = 0
    
    if y == 1:
        log_loss = -1 / x_L
    elif y == -1:
        log_loss = 1 / (1 - x_L)
        
    return log_loss
     
def pred(x_n, weights):   
    X = []
    S = []
    X, S = forwardPropagation(x_n, weights)
    
    #Classifying the arguments using 0.5 as the threshold
    if X[-1] >= 0.5:
        return 1
    else:
        return -1
    
def confMatrix(X_train, y_train, w):
    #Init output as a 2x2 matrix
    output = np.zeros((2, 2))         
    
    for i in range(X_train.shape[0]):
        f = pred(X_train[i], w)    
        if f > 0:       
            #Check points that are classified as 1
            if y_train[i] == 1:       
                output[1][1] += 1       
            if y_train[i] == -1:   
                output[0][1] += 1                              
        else:    
            #Check points that are classified as -1
            if y_train[i] == -1:              
                output[0][0] += 1       
            if y_train[i] == 1:       
                output[1][0] += 1  

    return output.astype(int)
    
def plotErr(e, epochs):
    #Plot the errors on a graph
    epoch = []
    for i in range(epochs):
            epoch.append(i + 1)
    plt.plot(epoch, e)
    plt.show
    
def test_SciKit(X_train, X_test, Y_train, Y_test):
    #Training and evaluating the scikit neural network
    scikit = MLPClassifier(hidden_layer_sizes = (300, 100), alpha = 10**-5, solver = 'sgd', random_state = 1)
    scikit.fit(X_train, Y_train)
    y_predict = scikit.predict(X_test)
    
    return confusion_matrix(Y_test, y_predict)

def test():
    X_train, y_train = load_iris(return_X_y = True)
    X_train, X_test, y_train, y_test = train_test_split(X_train[50:], y_train[50:], test_size = 0.2)
    
    for i in range(80):
        if y_train[i] == 1:
            y_train[i] = -1
        else:
            y_train[i] = 1
    for j in range(20):
        if y_test[j] == 1:
            y_test[j] = -1
        else:
            y_test[j] = 1
        
    err, w = fit_NeuralNetwork(X_train, y_train, 1e-2, [30, 10], 100)
   
    plotErr(err, 100)
    
    cM = confMatrix(X_test, y_test, w)
    
    sciKit = test_SciKit(X_train, X_test, y_train, y_test)
    
    print("Confusion Matrix is from Part 1a is: ", cM)
    print("Confusion Matrix from Part 1b is:", sciKit)

test()

'''
The matrices of our implementation of the neural network is different than the scikit-learn matrices at 100 epochs, further testing
with larger epochs such as 10000 shows that it becomes closer in-line with scikit-learn.  This might be caused by the differences 
with the alpha and hidden layer between our implementation and the scikit-learn.
The errors have a noticeable decrease through the first 20 epochs, but after that there is very small decrease 
in errors for the rest of the run.
Changing the number of layers and nodes would have an impact but it shouldn't be that big.
Changing the activation function from ReLu into a Sigmoid function might make a difference as it would change the
computation of the hidden layers in the forward and backward algorithms.
'''