from NN import NeuralNetwork
import numpy as np

layers = [2,3,3,1]
nn = NeuralNetwork(layers)

print(nn.affine(np.array((1,2)).reshape(2,1), 0))
