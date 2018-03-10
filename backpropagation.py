import numpy as np
from NN_model import *
from models import *
from PyNNfunctions import *

def compute_backprop(x, y, NN, learning_rate=0.1, activation=sigmoid, output_activation=identity):
    '''
    x = training input (list)
    y = training label (list)
    '''
    # If we're dealing with a sigmoidal neural network...
    if (activation == sigmoid) and (output_activation == identity):
        # Notation: https://brilliant.org/wiki/backpropagation/#formal-definition
        y = np.array(y)
        y_hat = np.array(NN.evaluate(x))
        inputLayer = NN._inputLayer()
        hiddenLayers = NN._hiddenLayers()
        outputLayer = NN._outputLayer()

        allLayers = NN._allLayers()

        # Create the (identical) delta values for the output layer...
        diff_array = np.array(y - y_hat)
        # (Updated from the brilliant.org work to allow n-ary classification by taking norms to
        # produce a scalar value)
        # Initialize the first round of delta values
        currentLayer = allLayers[-1]
        currentLayer.Deltas = np.sqrt(np.dot(diff_array, diff_array))

        # Since we're working _backwards_ through the hidden layers, we start at the last layer,
        # and step backwards to the first hidden layer. Note that since we are referencing the
        # previous layer at each step, we need to stop before we hit the 0'th hidden layer, since
        # that would attempt to access the -1'st hidden layer (i.e. the input layer), which isn't
        # in the hiddenLayers list. Note that this all could be rewritten nicer if I didn't separate
        # into input, hidden and output layers, and instead just worked with one big list of "layers".
        for i in range(len(allLayers)-2, 1, -1):

            ### BEGIN TESTING OF REARRANGEMENT ###
            currentLayer = allLayers[i]
            previousLayer = allLayers[i-1]
            nextLayer = allLayers[i+1]
            # Get the nextLayer.Weights...
            nextLayer.Weights = nextLayer.getWeights()
            print('nextLayer.Weights = {}'.format(nextLayer.Weights))
            print('nextLayer.Deltas = {}'.format(nextLayer.Deltas))
            # ...and dot them with the deltas from the previous layer.
            currentLayer.Deltas = np.dot(nextLayer.Deltas, nextLayer.Weights)
            print('currentLayer.Deltas = {}'.format(currentLayer.Deltas))
            # Get output from the previous layer
            previousLayer.Ohs = np.array(previousLayer.getVals())
            print('previousLayer.Ohs = {}'.format(previousLayer.Ohs))
            # Component-wise multiply them with the currentLayer.Deltas to get the gradients
            currentLayer.Gradients = []
            for deltas in currentLayer.Deltas:
                currentLayer.Gradients.append(previousLayer.Ohs * currentLayer.Deltas)
            print('currentLayer.Gradients = {}'.format(currentLayer.Gradients))

            for node in currentLayer.nodes:
                node_oldWeights = node.getWeight()
                node_updatedWeights = node_oldWeights - learning_rate*currentLayer.Gradients[i]
                node.setWeight(node_updatedWeights)
                i = i + 1

            ### END TESTING OF REARRANGEMENT ###
