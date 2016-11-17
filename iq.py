import time

from ANN_IQ.util import *

# Wildcard import defines cost and activation functions
# also imports numpy from the above namespace!

# Define hyperparameters according to the first part of
# M. Nielsen's neural network tutorial:
# http://neuralnetworksanddeeplearning.com/

LRATE = 3.
BSIZE = 20
EPOCHS = 30

# Instanciate activation and cost function wrappers
sigmoid = Sigmoid()
# tanh = Tanh()
# relu = ReLU
mse = MSE()
# xent = Xent()

mnistpath = "./mnist.pkl.gz"


class Layer:
    """
    Fully connected feed forward layer (aka. the Dense Layer)
    """

    def __init__(self, inputs, neurons):
        """
        Parameter initialization is done according to LeCun et al., 1998
        We simply ignore the fact that the initialization scheme is optimized
        for tanh layers... (we are going to use sigmoid layers)

        :param: inputs: the number of input nodes
        :param: neurons: the number of neurons
        """

        self.weights = np.random.randn(inputs, neurons) / np.sqrt(inputs)
        self.biases = np.zeros(neurons)
        self.actfn = sigmoid
        self.activations = None

    def get_weights(self):
        """Return the flattened weight matrices as 1D-vectors"""
        return self.weights.ravel()

    def set_weights(self, ws):
        """Fold the ws weight vector to a matrix"""
        self.weights = ws.reshape(self.weights.shape)


class Model:
    """
    Sequential (feed-forward) type of artificial neural network,
    Architecture is the Multilayer Perceptron (MLP).
    https://en.wikipedia.org/wiki/Multilayer_perceptron
    """

    def __init__(self):
        """
        This network is a concrete implementation with a
        (784-30-10) architecture to classify the MNIST
        reference data
        """

        self.hidden = Layer(784, 30)
        self.output = Layer(30, 10)
        self.inputs = None  # network input (X) goes here
        self.lrate = LRATE

        self.cost = mse

    def feedforward(self, X):
        """Produces output from the input 2D array (X)"""
        self.inputs = np.copy(X)

        self.hidden.activations = X @ self.hidden.weights
        self.hidden.activations += self.hidden.biases
        self.hidden.activations = self.hidden.actfn(self.hidden.activations)

        self.output.activations = self.hidden.activations @ self.output.weights
        self.output.activations += self.output.biases
        self.output.activations = self.output.actfn(self.output.activations)

        return self.output.activations

    def backpropagation(self, Y, get_grads=False):
        """Calculates the gradients and updates the parameters"""
        # size of the mini-batch, needed for the per-minibatch
        # gradient averaging step
        m = Y.shape[0]
        eta = self.lrate / m  # can be incorporated into learning rate

        # calculate the output layer's error
        delta_o = mse.derivative(Y, self.output.activations) * \
            self.output.actfn.derivative(self.output.activations)
        # backpropagate the error to the hidden layer
        delta_h = delta_o @ self.output.weights.T * \
            self.output.actfn.derivative(self.hidden.activations)

        # calculate weight gradients
        grad_Wo = self.hidden.activations.T @ delta_o
        grad_Wh = self.inputs.T @ delta_h

        if get_grads:
            # the parameter update is skipped and the calculated gradients
            # are returned as a single flattened 1D-array
            # used in self.gradient_check()
            return np.concatenate([grad_Wh.ravel(), grad_Wo.ravel()]) / m

        # update the parameters according to Stochastic Gradient Descent
        # w := w - dMSE/dw; b := b - dMSE/db
        self.output.weights -= grad_Wo * eta
        self.output.biases -= delta_o.sum(axis=0) * eta
        self.hidden.weights -= grad_Wh * eta
        self.hidden.biases -= delta_h.sum(axis=0) * eta

        # Gradient of the biases:
        # -----------------------
        # from z = X @ W + b
        # and a = sigm(z)
        #
        # dMSE/db = dMSE/da * da/dz * dz/db
        # dMSE/da = a - y
        # da/dz is the sigmoid's derivative
        # da/dz = sigm(z) * (1 - sigm(z)) = a * (1 - a)!
        # dz/db = 1 (remember, its a partial derivative)
        #
        # So the bias' gradient = dMSE/da * da/dz * 1,
        # which equals delta_o (summed over the batch and averaged by m)
        # it is calculated similarily in hidden layers:
        # delta_h, summed over the batch, averaged by m.

    def fit(self, X, Y, no_epochs, batch_size, validation=()):
        """This method coordinates the learning process"""
        N = X.shape[0]  # nuber of total learning samples

        for epoch in range(1, no_epochs + 1):
            print("Epoch:", epoch)
            X, Y = shuffle_data(X, Y)

            costs = []  # accumulate costs for later averaging
            for batch_start in range(0, N, batch_size):
                batch_end = batch_start + batch_size
                batchX = X[batch_start:batch_end]
                batchY = Y[batch_start:batch_end]

                self.feedforward(batchX)
                self.backpropagation(batchY)

                cost = self.cost(batchY, self.output.activations)
                costs.append(cost)

                print("\rCost:  {0:.5f}".format(np.mean(costs)), end="")

            if validation:
                nvalid = validation[0].shape[0]
                learning_accuracy = self.evaluate(X[:nvalid], Y[:nvalid])
                testing_accuracy = self.evaluate(*validation)
                print("\tAccuracies: on learning: {0:.2%}\ton testing: {1:.2%}"
                      .format(learning_accuracy, testing_accuracy))
            else:
                print()

    def predict(self, X):
        """Converts the raw predictions into actual numbers"""
        raw_predictions = self.feedforward(X)
        return np.argmax(raw_predictions, axis=1)

    def evaluate(self, X, Y):
        """Evaluates the network's prediction accuracy"""
        preds = self.predict(X)  # This is a vector of numbers
        targets = np.argmax(Y, axis=1)  # so is this
        eq = np.equal(preds, targets)  # bool vector: True, where equal, False otherwise
        no_right_predictions = eq.sum()
        all_predictions = len(eq)
        # we get the rate of right predictions (the accuracy)
        return no_right_predictions / all_predictions

    def get_weights(self):
        """
        Get the flattened weights from the layers and return them as an
        1D-array
        """
        return np.concatenate([self.hidden.get_weights(),
                               self.output.get_weights()])

    def set_weights(self, ws):
        """Set layer weights. ws is an 1D-array (vector)"""
        size = self.hidden.weights.size
        self.hidden.set_weights(ws[:size])
        self.output.set_weights(ws[size:])

    def gradient_check(self, X, y, epsilon=1e-5):
        """
        Estimate gradients by the numerical differentiation formula:

        cost(w-epsilon) - cost(w+epsilon)
        --------------------------------- ~= df/dw, if epsilon is sufficiently small
                    2*epsilon

        If backprop was implemented correctly, the relative error between
        the numerical and analytical gradients should be minimal (~ 1e-7)
        Biases are not (yet?) included!
        """

        # Nested functions are only used for better readability

        def get_numerical_gradients():
            ws = self.get_weights()
            perturbation = np.zeros_like(ws, dtype=float)
            grads = np.copy(perturbation)

            rounds = str(len(ws))
            for i in range(len(ws)):  # This is done for every weight
                print("\rCalculating numerical gradients... {0:>{width}} / {1}"
                      .format(i + 1, rounds, width=len(rounds)), end="")

                perturbation[i] += epsilon

                self.set_weights(ws + perturbation)
                cost_plus = self.cost(self.feedforward(X), y)
                self.set_weights(ws - perturbation)
                cost_minus = self.cost(self.feedforward(X), y)

                grads[i] = (cost_plus - cost_minus)
                perturbation[i] = 0.0

            print()
            return grads / (2 * epsilon)

        def get_relative_error_and_diffs():
            norm = np.linalg.norm  # alias this function (the vector norm)
            diff = analytical - numerical
            relative_error = norm(diff) / max(norm(analytical), norm(numerical))

            return relative_error

        def print_evaluation(er):
            print("Gradient check:", end=" ")
            if er < 1e-6:
                print("PASSED!", end=" ")
            elif er < 1e-5:
                print("SUSPICIOUS!", end=" ")
            elif er < 1e-3:
                print("ERROR!", end=" ")
            else:
                print("FATAL ERROR!", end=" ")
            print("error =", er)

        self.fit(X, y, no_epochs=1, batch_size=20, validation=(X, y))

        numerical = get_numerical_gradients()
        analytical = self.backpropagation(y, get_grads=True)

        error = get_relative_error_and_diffs()

        print_evaluation(error)


def main():
    start = time.time()

    # Read data to memory. Data is split by 60k:10k.
    (mnistX, mnistY), (testX, testY) = pull_mnist_data(mnistpath)

    # Instanciate the network model
    model = Model()
    print("INITIAL PREDICTION ACCURACY:", model.evaluate(testX, testY))

    # Perform a gradient check against numerical gradients (backprop sanity check)
    model.gradient_check(mnistX[:20], mnistY[:20])

    # Fit model to data
    model.fit(X=mnistX, Y=mnistY, no_epochs=EPOCHS, batch_size=BSIZE,
              validation=(testX, testY))

    print("FINITE! Time required: {} s".format(int(time.time() - start)))


if __name__ == '__main__':
    main()
