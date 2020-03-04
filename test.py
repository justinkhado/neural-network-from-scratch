from NN import NeuralNetwork
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



layers = [4,3,3,1]
nn = NeuralNetwork(layers)

X_train = np.array([[1.,2.,3.,4.,5.],
                    [2.,3.,4.,1.,2.],
                    [4.,5.,1.,3.,2.],
                    [1.,2.,2.,2.,2.]])


cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']
length = len(cancer['feature_names'])

nn = NeuralNetwork([length,20,10,5,1])

X_train, X_test, y_train, y_test = train_test_split(X, y)

nn.fit(X_train, y_train, epochs=100)


