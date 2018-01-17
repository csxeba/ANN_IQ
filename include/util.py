import numpy as np


def mnist_to_learning_table(source: str):
    """The reason of this method's existance is that I'm lazy as ..."""
    import pickle
    import gzip

    # This hack below is needed because the supplied mnist.pkl.gz file is
    # downloaded from Nielsen's github repo:
    # https://github.com/mnielsen/neural-networks-and-deep-learning
    # His file was created under Python 2 with Windows type encoding
    # and I didn't convert it to UTF-8.
    # Of course by the time I wrote this comment, rationalizing my
    # laziness, I could've wrote a better, more readable implementation,
    # yet here I am still, leaving this method as it is, instanciating
    # private classes from the pickle module...
    # But alas, more disgusting things are happening all around us, so
    # why start to change the world by changing myself first?
    f = gzip.open(source)
    with f:
        u = pickle._Unpickler(f)
        u.encoding = "latin1"
        tup = u.load()
    f.close()

    # The original dataset is split so:
    # - 50k learning
    # - 10k testing (for hyperparameter optimization)
    # - 10k validation (for final model validation)
    # We unify them and reslice the complete dataset.
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype("float32")
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    return questions, targets


def pull_mnist_data(path, split=0.1):
    X, Y = mnist_to_learning_table(path)
    number_of_categories = len(np.unique(Y))

    # We create a one-hot representation from the classes in Y.
    onehot = np.eye(number_of_categories)[Y]
    # onehot is shaped thus:
    # (N, 10), where N is the number of examples, 10 is the 1-hot vector of the true label:
    # e.g. the number 6 is encoded so: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # values of Y are stored as 32 bit floats (double)

    # X is shaped thus:
    # (N, 784), where N is the number of examples, 784 = 28x28 represents the pixels.
    # values of X are normalized from 0.0 - 1.0 and stored as 32 bit floats (double)

    mnistX, mnistY = shuffle_data(X, onehot)
    if not split:
        return mnistX, mnistY

    number_of_samples = len(X)
    indices = np.arange(number_of_samples)
    np.random.shuffle(indices)

    number_of_testing_samples = int(number_of_samples * split)
    testing_data_indices = indices[:number_of_testing_samples]
    learning_data_indices = indices[number_of_testing_samples:]

    testX, testY = mnistX[testing_data_indices], mnistY[testing_data_indices]
    mnistX, mnistY = mnistX[learning_data_indices], mnistY[learning_data_indices]

    return mnistX, mnistY, testX, testY


def shuffle_data(X, Y):
    """Shuffles X and Y together"""
    shuffargs = np.arange(Y.shape[0])
    np.random.shuffle(shuffargs)
    return X[shuffargs], Y[shuffargs]


class Sigmoid:
    """
    The sigmoid (logistic) activation function [0, +1],
    mainly applied in output units, in hiddens in can be
    applied in certain situations.
    https://en.wikipedia.org/wiki/Sigmoid_function
    """
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        """Parameter <a> = sigmoid(z), see __call__(z)!"""
        return a * (1 - a)

    def __str__(self):
        return "Sigmoid"


class Tanh:
    """
    The hyperbolic tangent activation function [-1, +1],
    mostly applied in hidden neurons.
    https://en.wikipedia.org/wiki/Hyperbolic_function
    """
    def __call__(self, Z):
        return np.tanh(Z)

    def __str__(self): return "tanh"

    @staticmethod
    def derivative(A):
        return np.subtract(1.0, np.square(A))


class ReLU:
    """
    The Rectified Linear Unit, popular in deeper architectures,
    only used in hidden neurons.
    https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
    """
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.greater(z, 0) * z

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        """Parameter <a> = ReLU(z), see __call__(z)!"""
        return np.greater(a, 0).astype(float)


class MSE:
    """
    The Mean Squader Error cost function,
    applicable to any type of problem (classification or regression)
    https://en.wikipedia.org/wiki/Mean_squared_error
    """
    def __call__(self, targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((outputs - targets) ** 2) / targets.shape[0]

    @staticmethod
    def derivative(targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        # This is the true derivative of the MSE. Best to use with linear output neurons
        return outputs - targets


class BXent:
    """
    The binary cross-entropy cost function,
    applicable to 2-class and multiclass-multilabel classification problems
    https://en.wikipedia.org/wiki/Cross_entropy
    """

    def __call__(self, targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        # Basically -log(a) at the "right" neuron, -log(1 - a) otherwise
        # By right neuron I mean "the output neuron representing the correct label"
        return -np.sum(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs)) / targets.shape[0]

    @staticmethod
    def derivative(targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        # Simplified form -> only valid with sigmoid output neurons
        return outputs - targets


class CXent:
    """
    The categorical cross-entropy cost function,
    applicable to multiclass problems
    https://en.wikipedia.org/wiki/Cross_entropy
    """

    def __call__(self, targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        # Equivalent to the negative log likelihood
        return np.sum(targets * np.log(outputs)) / -targets.shape[0]

    @staticmethod
    def derivative(targets: np.ndarray, outputs: np.ndarray) -> np.ndarray:
        # Simplified from -> only valid with softmax output neurons
        return outputs - targets  # / ((outputs - 1) * outputs)
