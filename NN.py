import numpy as np 

class NeuralNetwork(object):
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
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(size_next, size_curr) * 0.01 for size_curr, size_next in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((size,1)) for size in layer_sizes[1:]]
        self.activations = []

    def fit(self, X_train, y_train, epochs):
        '''
        Takes in a training set and a number of epochs to train the neural network
        '''
        for i in range(epochs):
            self.activations = [X_train]
            self.feedforward(X_train)


    def feedforward(self, X_train):
        '''
        Takes in a training set, performs a forward propagation over the entire training set, and returns the predicted outputs

        Z: vector of the results of the affine function of all samples
        activations: matrix of all inputs of all samples where the inputs is the X_train initially and then the previous activations every instance after
        '''
        activations = X_train
        for layer in range(self.num_layers - 1):
            print(layer)
            Z = self.affine(activations, layer)
            activations = self.sigmoid(Z)
        
        return activations

    def backprop(self, y_pred, y_true):
        '''
        Takes in the predicted output and the true outputs and performs a backpropagation
        '''
        d_a = self.d_cost(y_pred, y_true)
        d_z = self.d_sigmoid()


    def affine(self, inputs, layer):
        '''
        Takes in the current layer a vector of inputs values (either the initial inputs or the activations of the last layer) 
        and returns the affine function
        '''
        return np.dot(self.weights[layer], inputs) + self.biases[layer]

    def sigmoid(self, z):
        '''
        Takes in the affine function and returns the sigmoid activation
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def cost(self, y_pred, y_true):
        '''
        Takes in the predictions calculated from the X_train and compares them to the true values from the y_test using the cross-entropy loss function
        '''
        return -(y_true*np.log10(y_pred) + (1-y_true)*np.log10(1-y_pred))

    def dw_affine(self, inputs, layer):
        '''
        Returns the partial derivative of the affine function with respect to the weights
        '''
        return inputs
    
    def db_affine(self, inputs, layer):
        '''
        Returns the partial derivative of the affine function with respect to the biases
        '''
        return 1

    def d_sigmoid(self, z):
        '''
        Returns the partial derivative of the sigmoid function with respect to the affine function
        '''
        return (np.exp(-z)) / ((1 + np.exp(-z))**2)

    def d_cost(self, y_pred, y_true):
        '''
        Returns the partial derivative of the cost function with respect to the activation function
        '''
        return ((1-y_true)/(1-y_pred)) - (y_true/y_pred)
    