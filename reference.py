import numpy as np

from include.util import pull_mnist_data


def get_keras_net():
    """Depends on Keras and Theano/Tensorflow"""
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import SGD

    model = Sequential([
        Dense(input_dim=784, output_dim=30, activation="sigmoid"),
        Dense(output_dim=10, activation="sigmoid")
    ])
    model.compile(optimizer=SGD(lr=3.0), loss="mse", metrics=["acc"])

    return model  # THIS IS A NETWORK OBJECT!


def get_theano_net():
    """Depends on theno"""
    import theano
    import theano.tensor as T

    print("Defining Theano computational graph...", end=" ")

    inputs = T.matrix(dtype="float32")
    targets = T.matrix(dtype="float32")

    m = inputs.shape[0].astype("float32")
    eta = np.array([3.0]).astype("float32")

    Wh = theano.shared(np.random.randn(784, 30).astype("float32"))
    Wo = theano.shared(np.random.randn(30, 10).astype("float32"))
    bh = theano.shared(np.zeros([30], dtype="float32"))
    bo = theano.shared(np.zeros([10], dtype="float32"))

    activation = T.nnet.sigmoid(inputs.dot(Wh) + bh)
    activation = T.nnet.sigmoid(activation.dot(Wo) + bo)

    prediction = T.argmax(activation, axis=1)

    cost = T.sum((targets - activation) ** 2) / m

    update_Wh = Wh - theano.grad(cost, wrt=Wh) * eta
    update_Wo = Wo - theano.grad(cost, wrt=Wo) * eta
    update_bh = bh - theano.grad(cost, wrt=bh) * eta
    update_bo = bo - theano.grad(cost, wrt=bo) * eta

    print("Done!\nCompiling theano functions...", end=" ")
    fit_batch = theano.function(inputs=[inputs, targets],
                                outputs=[cost],
                                updates=[(Wh, update_Wh),
                                         (Wo, update_Wo),
                                         (bh, update_bh),
                                         (bo, update_bo)])

    predict = theano.function(inputs=[inputs],
                              outputs=[prediction])
    print("Done!")
    return fit_batch, predict  # THESE ARE FUNCTION POINTERS!


def get_brainforged_net():
    """Depends on csxnet and csxdata, both available on my github
    (but there are no install scripts for them :) )"""
    from csxnet import Network
    from csxdata import CData, roots

    model = Network(CData(roots["misc"] + "mnist.pkl.gz", 0.18),
                    eta=3.0, lmbd1=0, lmbd2=0, mu=0, cost="mse")
    model.add_fc(30, activation="sigmoid")
    model.finalize_architecture(activation="sigmoid")

    return model  # THIS IS ALSO A NETWORK OBJECT!


def train_keras():
    print("-"*25)
    print("Keras Training!")
    network = get_keras_net()
    (X, y), validation = pull_mnist_data(mnistpath)
    network.fit(X, y, batch_size=20, nb_epoch=30, validation_data=validation)


def train_brainforge():
    print("-"*25)
    print("Brainforge training!")
    network = get_brainforged_net()
    network.fit(batch_size=20, epochs=30, monitor=["acc"])


def train_theano():
    print("-"*25)
    print("Theano training!")
    (X, y), validation = pull_mnist_data(mnistpath)

    fit_batch, predict = get_theano_net()
    N = X.shape[0]
    for epoch in range(1, 30):
        print("Epoch: {0} / {1}".format(epoch, 30))
        for i in range(0, N, 20):
            cost = fit_batch(X[i:i+20], y[i:i+20])[0]  # dat indexing...
            print("\rBatch {0:>{w}} / {1}; Cost: {2:.4f}"
                  .format(i+20, N, float(cost), w=len(str(N))), end="")
        print()


mnistpath = "./mnist.pkl.gz"

if __name__ == '__main__':
    train_keras()
    train_theano()
    train_brainforge()
