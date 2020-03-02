import numpy as np 

class NeuralNetwork:
    '''
    A simple Neural Network with any number of layers and samples that predicts a binary classification.
    '''
    def __init__(self, layer_sizes):
        '''
        Initializes the sizes, num_layers, weights, and biases.

        sizes: list of sizes of each layer
        num_layers: number of layers
        weights: list of weight matrices
        biases: list of bias vectors
        '''
        self.sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(size_curr, size_next) * 0.01 for size_curr, size_next in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((size,1)) for size in self.sizes[1:]]

    def affine(self, inputs, layer):
        '''
        Takes in the current layer a vector of inputs values (either the initial inputs or the activations of the last layer) 
        and calculates the affine function
        '''
        return np.dot(self.weights[layer].T, inputs) + self.biases[layer]
