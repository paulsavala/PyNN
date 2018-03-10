# PyNN
A feedforward neural network, built-from-scratch in Python. Helpful for understanding both the computational and structural side of a neural network.

# Why PyNN?
I built PyNN (pronounced "pine") to better understand how neural networks work. It's not hard to find other built-from-scratch neural networks online. However, these tend to be built to emphasize brevity. For example, see the [here](https://iamtrask.github.io/2015/07/12/basic-python-network/) or [here](http://www.wildml.com/2015/09/implementing-a-neural-network-from-scratch/). While there's inherently nothing *wrong* with these, if you don't already understand how a feedforward neural network works, looking at those programs won't help you one bit. 

Conversely, you can work with [Keras](https://keras.io/) or [Tensorflow](https://www.tensorflow.org/). These are both incredible tools, but they abstract away much of what's going on in the background. Therefore they too make it difficult to get a grasp of what's happening.

I created PyNN to bridge the gap between understanding the mathematics (i.e. computations) and the architecture of a basic feedforward neural network. If you've begun reading about neural networks and think you have a grasp of what's going on, I encourage you to dive into the code. 

# What can it do?
The following features are fully-implemented:
* Quickly and easily create a feedforward neural network in just three lines of code:
```python
# Instantiate a new neural network with an input layer consisting of 
# 5 nodes, 3 hidden layers each with 10 nodes, and an output layer 
# with 1 node (binary classification)
NN = feedForwardNN(5, (10, 3), 1)

# Build the connectors between all layers
NN.construct()

# Initialize the weights and biases, sampled from a normal Gaussian distribution
NN.initialize()

# You're ready to go!
```
* Evaluate the output at a given input:
```python
x = np.array([1, 5, 3.4, 7, -1])
NN.evaluate(x)
```

* View weights and biases to get a feel for what's happening:
```python
NN.getWeights()
```

# What am I still working on?
* Backpropagation. This is mostly complete, but needs some tuning.
* Updating of biases. Weights updating through backpropagation is mostly complete, but biases still need to be added.
