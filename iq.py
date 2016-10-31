import time

import numpy as np

from ANN_IQ.util import Sigmoid, MSE, pull_mnist_data, shuffle_data


class Model:
    def __init__(self):
        # Instanciate parameters according to LeCunn et al., 1998
        # Weights and biases of hidden neurons
        self.Wh = np.random.randn(784, 60) / np.sqrt(784.)
        self.bh = np.zeros([60])

        # Weights and biases of output neurons
        self.Wo = np.random.randn(60, 10) / np.sqrt(60.)
        self.bo = np.zeros([10])

        self.ah = None  # activation of hiddens
        self.ao = None  # activation of outputs

        self.inputs = None  # network input

        self.lrate = LRATE

    def feedforward(self, X):
        self.inputs = X
        self.ah = X @ self.Wh
        self.ah += self.bh
        self.ah = sigmoid(self.ah)

        self.ao = self.ah @ self.Wo
        self.ao += self.bo
        self.ao = sigmoid(self.ao)

        return self.ao

    def predict(self, X):
        output = self.feedforward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, Y):
        preds = self.predict(X)
        targets = np.argmax(Y, axis=1)
        eq = np.equal(preds, targets)
        return eq.sum() / len(eq)

    def backpropagation(self, Y):
        m = len(Y)  # minibatch size for gradient averaging
        eta = self.lrate / m

        cost = mse(Y, self.ao)

        # backpropagate the errors
        delta_o = mse.derivative(Y, self.ao) * sigmoid.derivative(self.ao)
        # delta_o = (self.ao - Y) * self.ao * (1 - self.ao)
        delta_h = delta_o @ self.Wo.T * sigmoid.derivative(self.ah)
        # delta_h = delta_o @ self.Wo.T * self.ah * (1 - self.ah)

        # calculate weight gradients
        grad_Wo = self.ah.T @ delta_o
        grad_Wh = self.inputs.T @ delta_h

        # descend on gradients
        self.Wo -= grad_Wo * eta
        self.bo -= delta_o.sum(axis=0) * eta
        self.Wh -= grad_Wh * eta
        self.bh -= delta_h.sum(axis=0) * eta

        return cost

    def fit(self, X, Y, no_epochs, batch_size, validation):
        N = X.shape[0]
        assert N == Y.shape[0]

        for epoch in range(1, no_epochs+1):
            print("Epoch:", epoch)
            X, Y = shuffle_data(X, Y)
            costs = []
            for batch_start in range(0, N, batch_size):
                batchX = X[batch_start:batch_start+batch_size]
                batchY = Y[batch_start:batch_start+batch_size]

                self.feedforward(batchX)
                costs.append(self.backpropagation(batchY))

                print("\rCost:  {}".format(np.mean(costs)), end="")

            print()
            if validation:
                print("LAcc:  {}".format(self.evaluate(X[:10000], Y[:10000])))
                print("TAcc:  {}".format(self.evaluate(*validation)))

start = time.time()
mnistpath = "/data/Prog/data/misc/mnist.pkl.gz"

# Instanciate functions
sigmoid = Sigmoid()
mse = MSE()

# Define hyperparameters
LRATE = 0.01
BSIZE = 20
EPOCHS = 30

# Read data to memory
(mnistX, mnistY), (testX, testY) = pull_mnist_data(mnistpath)

# Build network model
model = Model()
print("INITIAL ACC:", model.evaluate(testX, testY))

# Fit model to data
model.fit(X=mnistX, Y=mnistY, no_epochs=EPOCHS, batch_size=BSIZE,
          validation=(testX, testY))

print("FINITE! Time required: {} s".format(int(time.time()-start)))
