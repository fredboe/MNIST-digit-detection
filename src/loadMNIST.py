import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random
import numpy as np


def loadMNIST():
    """
        returns:
                x_train, y_train, x_test, y_test -> mnist data from tensorflow
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # updates the data to number between 0 and 1
    x_train = x_train/255
    x_test = x_test/255
    # creates an 1-dimensional array out of the data
    x_train = [sample.reshape(1, 784)[0] for sample in x_train]
    x_test = [sample.reshape(1, 784)[0] for sample in x_test]
    # creates an vector out of the label
    y_train = [add_one(y) for y in y_train]
    y_test = [add_one(y) for y in y_test]
    print("mnist loaded")
    return (x_train, y_train), (x_test, y_test)


def add_one(index):
    """
        parameters:
                    index: int between 0 and 9 (label)
        returns:
                array with a length of 10: 1 at the label else 0
    """
    arr = np.zeros(10)
    arr[index] = 1.0
    return arr


def plotMNIST_random():
    """
        plots a random image of the train images
    """
    index = random.randint(0, 59999)
    (x_train, y_train), (x_test, y_test) = loadMNIST()
    print("Image number {index}. It is a {number}".format(index=index, number=y_train[index]))
    plt.imshow(x_train[index], cmap="Greys")
    plt.show()


def plotMNIST(index):
    """
        plots an image of the x_test images
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.imshow(x_test[index], cmap="Greys")
    plt.show()
