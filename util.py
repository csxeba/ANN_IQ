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
    # private classes from of the pickle module...
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
    # We unify them and return the complete dataset.
    questions = np.concatenate((tup[0][0], tup[1][0], tup[2][0]))
    questions = questions.astype("float32", copy=False)
    targets = np.concatenate((tup[0][1], tup[1][1], tup[2][1]))
    return questions, targets


def pull_mnist_data(path):
    Xs, Ys = mnist_to_learning_table(path)
    N = Ys.shape[0]

    # We create a one-hot representation from the classes is Y.
    onehot = np.zeros((N, 10), dtype="float32")
    for i, y in enumerate(Ys):
        onehot[i, y] += np.asscalar(np.array([1.0]))
    # onehot is shaped thus:
    # (N, 10), where N is the number of examples, 10 is the 1-hot vector of the true label:
    # e.g. the number 6 is encoded so: [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    # values of Y are stored as 32 bit floats (double)

    # X is shaped thus:
    # (N, 784), where N is the number of examples, 784 = 28x28 represents the pixels.
    # values of X are normalized from 0.0 - 1.0 and stored as 32 bit floats (double)

    mnistX, mnistY = shuffle_data(Xs, onehot)
    testX, testY = mnistX[:10000], mnistY[:10000]
    mnistX, mnistY = mnistX[10000:], mnistY[10000:]

    return (mnistX, mnistY), (testX, testY)


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
    def __call__(self, y: np.ndarray, a: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((a - y)**2) / y.shape[0]

    @staticmethod
    def derivative(y: np.ndarray, a: np.ndarray) -> np.ndarray:
        return a - y

    def __str__(self):
        return "MSE"


class Xent:
    """
    The cross-entropy cost function,
    applicable only to classification problems (like MNIST)
    https://en.wikipedia.org/wiki/Cross_entropy
    """

    def __call__(self, y: np.ndarray, a: np.ndarray) -> np.ndarray:
        # Basically -log(a) at the "right" neuron, -log(1 - a) otherwise
        # By right neuron I mean "the output neuron representing the correct label"
        return -np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) / y.shape[0]

    @staticmethod
    def derivative(y: np.ndarray, a: np.ndarray) -> np.ndarray:
        # The denominator is factored out, if used with sigmoid output units!
        return (y - a) / ((a - 1) * a)

    def __str__(self):
        return "MSE"
