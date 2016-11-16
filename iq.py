import time

from ANN_IQ.util import *


# Wildcard import defines cost and activation functions
# also import numpy from the above namespace!

# Define hyperparameters
LRATE = 3.
BSIZE = 10
EPOCHS = 60


class Layer:

    def __init__(self, inputs, neurons):
        self.weights = np.random.randn(inputs, neurons)
        self.biases = np.zeros(neurons)

        self.activations = None

    def get_weights(self):
        return self.weights.ravel()

    def set_weights(self, ws):
        self.weights = ws.reshape(self.weights.shape)


class Model:

    def __init__(self):
        # Instanciate parameters according to LeCunn et al., 1998
        # Weights and biases of hidden neurons

        self.hidden = Layer(784, 30)
        self.output = Layer(30, 10)
        self.inputs = None  # network input (X) goes here
        self.lrate = LRATE

    def feedforward(self, X):
        self.inputs = X

        self.hidden.activations = X @ self.hidden.weights
        self.hidden.activations += self.hidden.biases
        self.hidden.activations = sigmoid(self.hidden.activations)

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

    def backpropagation(self, Y, get_grads=False):

        cost = mse(Y, self.output.activations)

        # backpropagate the errors
        delta_o = mse.derivative(Y, self.output.activations) * \
                  sigmoid.derivative(self.output.activations)

        delta_h = delta_o @ self.output.weights.T * \
                  sigmoid.derivative(self.hidden.activations)

        # calculate weight gradients
        grad_Wo = self.hidden.activations.T @ delta_o
        grad_Wh = self.inputs.T @ delta_h

        if get_grads:
            return np.concatenate([grad_Wh.ravel(), grad_Wo.ravel()])

        # descend on gradients
        m = Y.shape[0]  # batch size for gradient averaging
        eta = self.lrate / m

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

                print("\rCost:  {0:.5f}".format(np.mean(costs)), end="")

            if validation:
                print("\tLAcc: {0:.2%}\tTAcc: {0:.2%}"
                      .format(self.evaluate(X[:10000], Y[:10000]),
                              self.evaluate(*validation)))
            else:
                print()

    def get_weights(self):
        return np.concatenate([self.hidden.get_weights(), self.output.get_weights()])

    def set_weights(self, ws):
        size = self.hidden.weights.size
        self.hidden.set_weights(ws[:size])
        self.output.set_weights(ws[size:])

    def gradient_check(self, X, y, epsilon=1e-5):
        """
        Estimate gradients by the numerical differentiation formula:

        f(w-epsilon) - f(w+epsilon)
        --------------------------- ~= df/dw, if epsilon is sufficiently small.
                 2*epsilon

        If backprop was implemented correctly, the difference between the numerical
        and analytical gradients should be minimal (~ 1e-7)
        """

        def get_numerical_gradients():
            ws = self.get_weights()
            perturbation = np.zeros_like(ws, dtype=float)
            grads = np.copy(perturbation)

            rounds = str(len(ws))
            for i in range(len(ws)):
                print("\rCalculating numerical gradients... {0:>{w}} / {1}"
                      .format(i+1, rounds, w=len(rounds)), end="")

                perturbation[i] += epsilon

                self.set_weights(ws + perturbation)
                cost_plus = mse(self.feedforward(X), y)
                self.set_weights(ws - perturbation)
                cost_minus = mse(self.feedforward(X), y)

                grads[i] = (cost_plus - cost_minus)
                perturbation[i] = 0.0

            print()
            return grads / (2*epsilon)

        def get_relative_error_and_diffs():
            norm = np.linalg.norm
            diff = analytic - numeric
            relative_error = norm(diff) / max(norm(analytic), norm(numeric))

            size = self.hidden.weights.size
            hshape = self.hidden.weights.shape
            oshape = self.output.weights.shape
            hdiff = diff[:size].reshape(hshape)
            odiff = diff[size:].reshape(oshape)

            return relative_error, [hdiff, odiff]

        def print_evaluation():
            print("Gradient check:", end=" ")
            if error < 1e-6:
                print("PASSED!", end=" ")
            elif error < 1e-5:
                print("SUSPICIOUS!", end=" ")
            elif error < 1e-3:
                print("ERROR!", end=" ")
            else:
                print("FATAL ERROR!", end=" ")
            print("error =", error)

        self.fit(X, y, no_epochs=1, batch_size=20, validation=(X, y))

        numerical = get_numerical_gradients()
        analytical = self.backpropagation(y, get_grads=True)

        error, diffs = get_relative_error_and_diffs()

        print_evaluation()
        return error, diffs


start = time.time()
mnistpath = "D:/Data/misc/mnist.pkl.gz"

# Instanciate activation and cost function wrappers
sigmoid = Sigmoid()
mse = MSE()

# Read data to memory. Data is split by 60k:10k.
(mnistX, mnistY), (testX, testY) = pull_mnist_data(mnistpath)

# Instanciate the network model
model = Model()
print("INITIAL ACC:", model.evaluate(testX, testY))

# Perform a gradient check against numerical gradients (backprop sanity check)
model.gradient_check(mnistX[:20], mnistY[:20])

# Fit model to data
model.fit(X=mnistX, Y=mnistY, no_epochs=EPOCHS, batch_size=BSIZE,
          validation=(testX, testY))

print("FINITE! Time required: {} s".format(int(time.time() - start)))
