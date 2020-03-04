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
        self.weights = [np.random.randn(size_next, size_curr) * 0.01 for size_curr, size_next in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.zeros((size,1)) for size in layer_sizes[1:]]
        self.Z = []
        self.A = []

    def fit(self, X_train, y_train, epochs, learning_rate=0.1):
        '''
        Takes in a training set and a number of epochs to train the neural network
        '''
        X = X_train.T
        for i in range(epochs):
            self.feedforward(X)
            self.backprop(self.A[-1], y_train, learning_rate)

    def feedforward(self, X):
        '''
        Takes in a training set, performs a forward propagation over the entire training set, and returns the predicted outputs

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
        dL_da = self.d_cost(y_pred, y_true)
        for layer in range(self.num_layers - 2, -1, -1):
            da_dz = self.d_sigmoid(self.Z[layer])
            dz_dw = self.dw_affine(self.A[layer])
            dz_db = self.db_affine()

            for i in self.weights:
                print(i.shape)
            print()
            for i in self.A:
                print(i.shape)
            print()
            print(dL_da.shape, da_dz.shape, dz_db.shape)

            dL_dw = np.dot(dL_da * da_dz, dz_dw.T)
            dL_db = dL_da * da_dz * dz_db

            print()
            print(dL_dw.shape)

            self.weights[layer] -= learning_rate*dL_dw
            self.biases[layer] -= learning_rate*dL_db

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

    def cross_entropy(self, y_pred, y_true):
        '''
        Takes in the predictions calculated from the X_train and compares them to the true values from the y_test using cross-entropy
        '''
        return -(y_true*np.log10(y_pred) + (1-y_true)*np.log10(1-y_pred))

    def cost(self, y_pred, y_true):
        '''
        Returns the average cost of all the samples using the cross entropy loss function
        '''
        return (1/len(y_pred)) * sum(self.cross_entropy(y_pred, y_true))

    def dw_affine(self, inputs):
        '''
        Returns the partial derivative of the affine function with respect to the weights
        '''
        return inputs
    
    def db_affine(self):
        '''
        Returns the partial derivative of the affine function with respect to the biases
        '''
        return np.ones(1).reshape(1,1)

    def d_sigmoid(self, z):
        '''
        Returns the partial derivative of the sigmoid function with respect to the affine function
        '''
        return (np.exp(-z)) / ((1 + np.exp(-z))**2)

    def d_cost(self, y_pred, y_true):
        '''
        Returns the partial derivative of the average cost function with respect to the activation function
        '''
        return (1/len(y_pred))*(((1-y_true)/(1-y_pred)) - (y_true/y_pred))
    