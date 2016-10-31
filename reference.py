from ANN_IQ.util import pull_mnist_data


def get_keras_net():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    model = Sequential([
        Dense(input_dim=784, output_dim=30, activation="sigmoid"),
        Dense(input_dim=30, output_dim=10, activation="sigmoid")
    ])
    model.compile(optimizer=SGD(lr=0.5), loss="mse", metrics=["acc"])

    return model


def get_brainforged_net():
    from csxnet import Network
    from csxdata import CData, roots

    model = Network(CData(roots["misc"] + "mnist.pkl.gz", 0.18), 0.5, 0, 0, 0, cost="mse")
    model.add_fc(60, activation="sigmoid")
    model.finalize_architecture()

    return model

(X, y), validation = pull_mnist_data("mnist.pkl.gz")
network = get_keras_net()

network.fit(X, y, 20, 10, 1, validation_data=validation)
