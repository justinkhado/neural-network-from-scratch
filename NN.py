import numpy as np 

class NeuralNetwork(object):
    '''
    A simple Neural Network with any number of layers and samples that predicts a binary classification.
    '''
    def __init__(self, layer_sizes):
        '''
        sizes: list of sizes of each layer
        num_layers: number of layers
        weights: list of weight matrices
        biases: list of bias vectors
        Z: list of matrices of affine function values
        A: list of matrices of activations
        '''
        self.num_layers = len(layer_sizes)
        self.weights = [np.random.randn(size_next, size_curr)*0.01 for size_curr, size_next in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((size,1)) for size in layer_sizes[1:]]
        self.Z = []
        self.A = []

    def fit(self, X_train, y_train, epochs, learning_rate=0.1, verbose=True):
        '''
        Takes in a training set and a number of epochs to train the neural network
        '''
        X = X_train.T
        for i in range(epochs):
            
            if verbose:
                print(f'Epoch: {i+1}/{epochs}')
            self.feedforward(X)
            self.backprop(self.A[-1], y_train, learning_rate)

    def predict(self, X):
        '''
        Returns an array of predictions for each input
        '''
        self.feedforward(X.T)
        return self.A[-1]

    def feedforward(self, X):
        '''
        Takes in a training set and performs a forward propagation over the entire training set

        z: vector of the results of the affine function of all samples
        activations: matrix of all inputs of all samples where the inputs is the X initially and then the previous activations every instance after
        '''
        activations = X
        self.Z = []
        self.A = [X]

        for layer in range(self.num_layers - 1):
            z = self.affine(activations, layer)
            self.Z.append(z)
            activations = self.sigmoid(z)
            self.A.append(activations)
        
    def backprop(self, y_pred, y_true, learning_rate):
        '''
        Takes in the predicted output and the true outputs and performs a backpropagation
        '''
        num_samples = len(y_pred[0])
        dL_da = self.d_cost(y_pred, y_true)
        for layer in reversed(range(self.num_layers - 1)):
            dL_dz = np.multiply(dL_da, self.d_sigmoid(self.Z[layer]))
            dL_dw = (1.0/num_samples)*np.dot(dL_dz, self.A[layer].T)
            dL_db = (1.0/num_samples)*np.sum(dL_dz, axis=1, keepdims=True)
            dL_da = np.dot(self.weights[layer].T, dL_dz)

            self.weights[layer] -= learning_rate*dL_dw
            self.biases[layer] -= learning_rate*dL_db

    def affine(self, inputs, layer):
        '''
        Returns the affine function by calculating the sum of a vector of biases and dot product of a matrix of weights and the inputs   
        '''
        return np.dot(self.weights[layer], inputs) + self.biases[layer]

    def sigmoid(self, z):
        '''
        Returns the sigmoid activation with the affine function as an input
        '''
        return 1.0 / (1.0 + np.exp(-z))

    def cross_entropy(self, y_pred, y_true):
        '''
        Returns a vector consisting of costs of each sample calculated using the cross entropy loss function
        '''
        return -(y_true*np.log10(y_pred) + (1-y_true)*np.log10(1-y_pred))

    def cost(self, y_pred, y_true):
        '''
        Returns the average cost of all the samples using the cross entropy loss function
        '''
        num_samples = len(y_pred[0])
        return (1.0/num_samples) * np.sum(self.cross_entropy(y_pred, y_true))

    def d_sigmoid(self, z):
        '''
        Returns the partial derivative of the sigmoid function with respect to the affine function
        '''
        return (np.exp(-z)) / ((1.0 + np.exp(-z))**2)

    def d_cost(self, y_pred, y_true):
        '''
        Returns the partial derivative of the cross entropy cost function with respect to the activation function
        '''
        return ((1.0-y_true)/(1.0-y_pred)) - (y_true/y_pred)
