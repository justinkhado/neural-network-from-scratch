from NN import NeuralNetwork
import numpy as np

layers = [4,3,3,1]
nn = NeuralNetwork(layers)

X_train = np.array([[1.,2.,3.,4.,5.],
                    [2.,3.,4.,1.,2.],
                    [4.,5.,1.,3.,2.],
                    [1.,2.,2.,2.,2.]])

z = nn.affine(X_train, 0)
a = nn.sigmoid(z)

z = nn.affine(a, 1)
a = nn.sigmoid(z)

z = nn.affine(a, 2)
a = nn.sigmoid(z)

print(nn.feedforward(X_train))


