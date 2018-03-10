import numpy as np
from NN_model import *
from backpropagation import *

NN = feedForwardNN(2, (4,3), 1)
NN.construct()
NN.initialize()

x = np.array([1, 1])
y = np.array([1, 2, 3, 4, 5])
compute_backprop(x, y, NN)
