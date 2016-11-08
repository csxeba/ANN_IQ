import time

from ANN_IQ.util import *
# Wildcard import defines cost and activation functions
# also import numpy from the above namespace!

# Define hyperparameters
LRATE = 0.01
BSIZE = 20
EPOCHS = 30


class Layer:
    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs)
        self.biases = np.zeros(neurons)

        self.activations = None


class Model:
    def __init__(self):
        # Instanciate parameters according to LeCunn et al., 1998
        # Weights and biases of hidden neurons

        self.hidden = Layer(784, 60)
        self.output = Layer(60, 10)
        self.inputs = None  # network input
        self.lrate = LRATE

    def feedforward(self, X):
        self.inputs = X

        self.hidden.activations = X @ self.hidden.weights
        self.hidden.activations += self.hidden.biases
        self.hidden.activations = relu(self.hidden.activations)

        self.output.activations = self.hidden.activations @ self.output.weights
        self.output.activations += self.output.biases
        self.output.activations = sigmoid(self.output.activations)

        return self.output.activations

    def predict(self, X):
        output = self.feedforward(X)
        return np.argmax(output, axis=1)

    def evaluate(self, X, Y):
        preds = self.predict(X)
        targets = np.argmax(Y, axis=1)
        eq = np.equal(preds, targets)
        assert len(set(ar.shape[0] for ar in (X, Y, preds, targets, eq))) == 1
        return eq.sum() / len(eq)

    def backpropagation(self, Y):
        m = len(Y)  # minibatch size for gradient averaging
        eta = self.lrate / m

        cost = mse(Y, self.output.activations)

        # backpropagate the errors
        delta_o = mse.derivative(Y, self.output.activations) * \
            sigmoid.derivative(self.output.activations)

        delta_h = delta_o @ self.output.weights.T * \
            sigmoid.derivative(self.hidden.activations)

        # calculate weight gradients
        grad_Wo = self.hidden.activations.T @ delta_o
        grad_Wh = self.inputs.T @ delta_h

        # descend on gradients
        self.output.weights -= grad_Wo * eta
        self.output.biases -= delta_o.sum(axis=0) * eta
        self.hidden.weights -= grad_Wh * eta
        self.hidden.biases -= delta_h.sum(axis=0) * eta

        return cost

    def fit(self, X, Y, no_epochs, batch_size, validation):
        N = X.shape[0]
        assert N == Y.shape[0]

        for epoch in range(1, no_epochs + 1):
            print("Epoch:", epoch)
            X, Y = shuffle_data(X, Y)
            costs = []
            for batch_start in range(0, N, batch_size):
                batchX = X[batch_start:batch_start + batch_size]
                batchY = Y[batch_start:batch_start + batch_size]

                self.feedforward(batchX)
                cost = self.backpropagation(batchY)
                costs.append(cost)

                print("\rCost:  {}".format(np.mean(costs)), end="")

            print()
            if validation:
                print("LAcc:  {}".format(self.evaluate(X[:10000], Y[:10000])))
                print("TAcc:  {}".format(self.evaluate(*validation)))


start = time.time()
mnistpath = "D:/Data/misc/mnist.pkl.gz"

# Instanciate functions
sigmoid = Sigmoid()
mse = MSE()
relu = ReLU()

# Read data to memory
(mnistX, mnistY), (testX, testY) = pull_mnist_data(mnistpath)

# Build network model
model = Model()
print("INITIAL ACC:", model.evaluate(testX, testY))

# Fit model to data
model.fit(X=mnistX, Y=mnistY, no_epochs=EPOCHS, batch_size=BSIZE,
          validation=(testX, testY))

print("FINITE! Time required: {} s".format(int(time.time() - start)))
