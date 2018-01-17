import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Dense

from include.util import pull_mnist_data


XSHAPE = (28, 28)
YSHAPE = (10, 1)


def build_model(inshape, outshape):
    ann = Sequential([
        Dense(60, input_shape=inshape, activation="sigmoid"),
        Dense(outshape[0], activation="softmax")
    ])
    ann.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["acc"])
    return ann


def run_showcase():

    def run_visual_loop(iterations=5):
        title.set_text(titletmpl.format(model.evaluate(images, labels, verbose=0)[-1]))
        predictions = model.predict(images, verbose=0)
        for i, (im, lb, pr) in enumerate(zip(images, labels, predictions), start=1):
            xobj.set_data(im.reshape(*XSHAPE))
            yobj.set_data(lb.reshape(*YSHAPE))
            aobj.set_data(pr.reshape(*YSHAPE))
            for pred, annot in zip(pr, atxt):
                annot.set_text(f"{pred:>6.2%}")
            plt.pause(0.5)
            if i >= iterations:
                break

    lX, lY, images, labels = pull_mnist_data("Data/mnist.pkl.gz", split=0.1)

    model = build_model(images.shape[1:], labels.shape[1:])

    plt.ion()
    fig, (lx, mx, rx) = plt.subplots(1, 3)
    xobj = lx.imshow(np.zeros(XSHAPE), vmin=0., vmax=1., cmap="Greys")
    yobj = mx.imshow(np.zeros(YSHAPE), vmin=0., vmax=1., cmap="Greys")
    aobj = rx.imshow(np.zeros(YSHAPE), vmin=0., vmax=1., cmap="hot")
    atxt = [rx.annotate(f"{0.:>6.2%}", xy=(.5, i), va="center") for i in range(10)]
    titletmpl = "Current training accuracy: {:>6.2%}"
    title = plt.suptitle(titletmpl.format(model.evaluate(images, labels, batch_size=256, verbose=0)[-1]))

    mx.set_yticks(range(10))
    mx.set_xticks([0])
    rx.set_yticks(range(10))
    rx.set_xticks([0])

    lx.set_title("Input image")
    mx.set_title("Target Y")
    rx.set_title("Predicted Y")

    run_visual_loop()
    model.fit(lX, lY, batch_size=64, epochs=1, verbose=2)
    run_visual_loop()


if __name__ == '__main__':
    run_showcase()
