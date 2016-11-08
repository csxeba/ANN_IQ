import numpy as np


def pull_mnist_data(path):
    from csxdata.utilities.parsers import mnist_tolearningtable
    Xs, Ys = mnist_tolearningtable(path, fold=False)
    N = Ys.shape[0]

    onehot = np.zeros((N, 10), dtype="float32")
    for i, y in enumerate(Ys):
        onehot[i, y] += np.asscalar(np.array([1.0]))

    mnistX, mnistY = shuffle_data(Xs, onehot)
    testX, testY = mnistX[:10000], mnistY[:10000]
    mnistX, mnistY = mnistX[10000:], mnistY[10000:]

    return (mnistX, mnistY), (testX, testY)


def shuffle_data(X, Y):
    shuffargs = np.arange(len(Y))
    np.random.shuffle(shuffargs)
    return X[shuffargs], Y[shuffargs]


class Sigmoid:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(z))

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        return a * (1 - a)

    def __str__(self):
        return "Sigmoid"


class ReLU:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.greater(z, 0) * z

    @staticmethod
    def derivative(a: np.ndarray) -> np.ndarray:
        return np.greater(a, 0).astype(float)


class MSE:
    def __call__(self, y: np.ndarray, a: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((y - a)**2, axis=0)

    @staticmethod
    def derivative(y: np.ndarray, a: np.ndarray) -> np.ndarray:
        asd = y - a
        return asd

    def __str__(self):
        return "MSE"
