import numpy as np
from PyNNfunctions import *

class Node:
    def __init__(self):
        self.val = None

    def setVal(self, val):
        self.val = val

    def getVal(self):
        return self.val

class HiddenNode(Node):
    def __init__(self, weight=None, bias=None, activation=sigmoid):
        Node.__init__(self)
        self.weight = weight
        self.bias = bias
        self.activation = activation

    def getWeight(self):
        return self.weight

    def getBias(self):
        return self.bias

    def setWeight(self, w):
        self.weight = w

    def setBias(self, b):
        self.bias = b

    def setLinearVal(self, x):
        self.linear_val = np.dot(self.weight, x) + self.bias

    def setVal(self, x):
        self.setLinearVal(x)
        self.val = self.activation(self.linear_val)

    def getLinearVal(self):
        return self.linear_val

class InputNode(Node):
    def __init__(self, val=None):
        Node.__init__(self)
        self.val = val

class OutputNode(HiddenNode):
    def __init__(self, weight=None, bias=None, activation=identity):
        HiddenNode.__init__(self, weight, bias, activation)

class Layer:
    def __init__(self):
        self.nodes = []

    def addNode(self, node):
        self.nodes.append(node)

    def size(self):
        return len(self.nodes)

    def getNodes(self):
        return self.nodes

    def getVals(self):
        vals = []
        for node in self.nodes:
            vals.append(node.getVal())
        return vals

    def getLinearVals(self):
        linear_vals = []
        for node in self.nodes:
            linear_vals.append(node.getLinearVal())
        return linear_vals

    def setVals(self, x):
        i = 0
        if type(x) != np.ndarray:
            x = np.array(x)
        for node in self.nodes:
            node.setVal(x[i])
            i = i + 1

    def gaussianVals(self):
        gaussian_vals = [np.random.normal() for n in range(len(self.nodes))]
        self.setVals(self, np.random.normal)

class InputLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

class HiddenLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

    def gaussianWeights(self, inputLayerSize):
        for node in self.nodes:
            gaussian_weights = np.array([np.random.normal() for n in range(inputLayerSize)])
            node.setWeight(gaussian_weights)

    def gaussianBias(self):
        for node in self.nodes:
            gaussian_bias = np.random.normal()
            node.setBias(gaussian_bias)

    def initializeWeightsAndBias(self, inputLayerSize):
        self.gaussianWeights(inputLayerSize)
        self.gaussianBias()

    def activate(self, tailLayer):
        for node in self.nodes:
            node.setVal(tailLayer.getVals())

    def getWeights(self):
        node_weights = []
        for node in self.nodes:
            node_weights.append(node.getWeight())
        return node_weights

    def setWeights(self, w):
        i = 0
        if type(w) != np.ndarray:
            w = np.array(w)
        for node in self.nodes:
            node.setWeight(w[i])

class OutputLayer(Layer):
    def __init__(self):
        Layer.__init__(self)

    def gaussianWeights(self, inputLayerSize):
        # num = input layer size
        for node in self.nodes:
            gaussian_weights = np.array([np.random.normal() for n in range(inputLayerSize)])
            node.setWeight(gaussian_weights)

    def gaussianBias(self):
        for node in self.nodes:
            gaussian_bias = np.random.normal()
            node.setBias(gaussian_bias)

    def initializeWeightsAndBias(self, inputLayerSize):
        self.gaussianWeights(inputLayerSize)
        self.gaussianBias()

    def getWeights(self):
        node_weights = []
        for node in self.nodes:
            node_weights.append(node.getWeight())
        return node_weights

    def setWeights(self, w):
        i = 0
        if type(w) != np.ndarray:
            w = np.array(w)
        for node in self.nodes:
            node.setWeight(w[i])

    def activate(self, tailLayer):
        for node in self.nodes:
            node.setVal(tailLayer.getVals())

    def getProbs(self):
        vals = self.getVals()
        probs = softmax(vals)
        return probs

class Connector:
    def __init__(self, tailLayer, tipLayer):
        self.tailLayer = tailLayer
        self.tipLayer = tipLayer

    def getTailLayer(self):
        return self.tailLayer

    def getTipLayer(self):
        return self.tipLayer

    def setTailLayer(self, tailLayer):
        self.tailLayer = tailLayer

    def setTipLayer(self, tipLayer):
        self.tipLayer = tipLayer

    def activate(self):
        self.tipLayer.activate(self.tailLayer)
