import numpy as np

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


def get_theano_net():
    import theano
    import theano.tensor as T

    inputs = T.matrix(dtype="float32")
    targets = T.matrix(dtype="float32")

    eta = np.array([0.5]).astype("float32")

    Wh = theano.shared(np.random.randn(784, 60).astype("float32"))
    Wo = theano.shared(np.random.randn(60, 10).astype("float32"))
    bh = theano.shared(np.zeros([60], dtype="float32"))
    bo = theano.shared(np.zeros([10], dtype="float32"))

    activation = T.nnet.sigmoid(inputs.dot(Wh) + bh)
    activation = T.nnet.sigmoid(activation.dot(Wo) + bo)

    cost = T.sum((targets - activation) ** 2)

    update_Wh = Wh - theano.grad(cost, wrt=Wh) * eta
    update_Wo = Wo - theano.grad(cost, wrt=Wo) * eta
    update_bh = bh - theano.grad(cost, wrt=bh) * eta
    update_bo = bo - theano.grad(cost, wrt=bo) * eta

    training = theano.function(inputs=[inputs, targets],
                               outputs=[cost],
                               updates=[(Wh, update_Wh),
                                        (Wo, update_Wo),
                                        (bh, update_bh),
                                        (bo, update_bo)])
    return training


def get_brainforged_net():
    from csxnet import Network
    from csxdata import CData, roots

    model = Network(CData(roots["misc"] + "mnist.pkl.gz", 0.18), 0.5, 0, 0, 0, cost="mse")
    model.add_fc(60, activation="sigmoid")
    model.finalize_architecture()

    return model

mnistpath = "/data/Prog/data/misc/mnist.pkl.gz"

(X, y), validation = pull_mnist_data(mnistpath)
network = get_keras_net()

network.fit(X, y, batch_size=20, nb_epoch=30, validation_data=validation)
