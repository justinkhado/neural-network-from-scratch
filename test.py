from NN import NeuralNetwork
from Normalizer import Normalizer
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split



cancer = load_breast_cancer()
X = cancer['data']
y = cancer['target']
length = len(cancer['feature_names'])

nn = NeuralNetwork([length,30,20,10,5,1])

X_train, X_test, y_train, y_test = train_test_split(X, y)

normalize = Normalizer()
normalize.fit(X_train)
X_train = normalize.transform(X_train)
X_test = normalize.transform(X_test)

nn.fit(X_train, y_train, epochs=1000, verbose=False)
predictions = nn.predict(X_test)
print(nn.cost(predictions, y_test))






