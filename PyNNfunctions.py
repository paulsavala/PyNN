import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def der_sigmoid(x):
    '''
    Derivative of the sigmoid function
    '''
    ds = sigmoid(x)*(1-sigmoid(x))
    return ds

def identity(x):
    return x

def softmax(x):
    s = [np.exp(x[k]) / np.sum(np.exp(x)) for k in range(len(x))]
    return s

def mse(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    N = len(y)
    return 1/(2*N)*np.sum(np.square(y-y_hat))

sigmoid = np.vectorize(sigmoid)
def_sigmoid = np.vectorize(der_sigmoid)
identity = np.vectorize(identity)
