import numpy as np
from PyNNfunctions import *
from models import *

class feedForwardNN:
    def __init__(self, inputLayerSize, hiddenLayersSize, outputLayerSize):
        '''
        Begin construction of a blank feed forward NN.
        (Int) inputLayerSize is the number of input nodes
        (Tuple) hiddenLayersSize = (m, n) with m nodes in each
            of n layers
        By assumption, the size of the output layer must be the same
        as the size of the input layer, so it is not requested as input.
        '''
        self.inputLayerSize = inputLayerSize
        self.numHiddenNodes = hiddenLayersSize[0]
        self.numHiddenLayers = hiddenLayersSize[1]
        self.outputLayerSize = outputLayerSize

    def construct(self):
        # Construct the input layer
        inputNodes = [InputNode() for n in range(self.inputLayerSize)]
        inputLayer = InputLayer()
        self.inputLayer = inputLayer
        while len(inputNodes) > 0:
            inputLayer.addNode(inputNodes.pop())

        # We will need to construct (potentially) multiple hidden layers. Prep this
        # by first finding how many needed.
        hiddenLayersList = []
        numHiddenNodesPerLayer = self.numHiddenNodes
        numHiddenLayers = self.numHiddenLayers

        # Construct the requested number of hidden layers
        n = 1
        while n <= self.numHiddenLayers:
            hiddenNodes = [HiddenNode() for _ in range(numHiddenNodesPerLayer)]
            hiddenLayer = HiddenLayer()
            while len(hiddenNodes) > 0:
                hiddenLayer.addNode(hiddenNodes.pop())
            hiddenLayersList.append(hiddenLayer)
            n = n + 1

        # Begin creating connectors. Start by creating a connector from the input layer to the first
        # hidden layer
        connectors = []
        connectors.append(Connector(inputLayer, hiddenLayersList[0]))

        # Now, create the rest of the requested hidden layer connectors
        i = 0
        while i < self.numHiddenLayers - 1:
            current_hidden = hiddenLayersList[i]
            next_hidden = hiddenLayersList[i+1]
            connectors.append(Connector(current_hidden, next_hidden))
            i = i + 1

        # Create the output layer
        outputNodes = [OutputNode() for n in range(self.outputLayerSize)]
        outputLayer = OutputLayer()

        j = 0
        while j < self.outputLayerSize:
            outputLayer.addNode(outputNodes[j])
            j = j + 1

        # Connect the final hidden layer to the output layer
        last_hidden = hiddenLayersList[-1]
        connectors.append(Connector(last_hidden, outputLayer))

        self.outputLayer = outputLayer
        self.hiddenLayers = hiddenLayersList
        self.allLayers = [self.inputLayer] + self.hiddenLayers + [self.outputLayer]

        self.connectors = connectors

    def initialize(self):
        outputLayer = self.outputLayer
        outputLayer.initializeWeightsAndBias(self.numHiddenNodes)

        # Initialize first hidden layer (different number of weights than further hidden layers)
        firstHiddenLayer = self.hiddenLayers[0]
        firstHiddenLayer.initializeWeightsAndBias(self.inputLayerSize)

        # Initialize remaining hidden layers
        if self.numHiddenLayers > 1:
            for layer in self.hiddenLayers[1:]:
                layer.initializeWeightsAndBias(self.numHiddenNodes)

    def evaluate(self, x):
        self.inputLayer.setVals(x)
        if self.connectors == None:
            return 'Network not initialized. Run .initialize() first'

        for connector in self.connectors:
            connector.activate()
        return self.outputLayer.getVals()

    def getWeights(self):
        outputWeights = self.outputLayer.getWeights()
        hiddenWeights = []
        i = 1
        for layer in self.hiddenLayers:
            hiddenWeights.append('Layer {}: {}'.format(i, layer.getWeights()))
            i = i + 1
        hiddenWeights = np.array(hiddenWeights)
        return hiddenWeights, outputWeights

    def _inputLayer(self):
        return self.inputLayer

    def _hiddenLayers(self):
        return self.hiddenLayers

    def _outputLayer(self):
        return self.outputLayer

    def _allLayers(self):
        return self.allLayers
