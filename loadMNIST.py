import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random


def loadMNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("mnist loaded")
    return (x_train, y_train), (x_test, y_test)


def plotMNIST_random():
    index = random.randint(0, 59999)
    (x_train, y_train), (x_test, y_test) = loadMNIST()
    print("Image number {index}. It is a {number}".format(index=index, number=y_train[index]))
    plt.imshow(x_train[index], cmap="Greys")
    plt.show()


def plotMNIST(training_data):
    plt.imshow(training_data, cmap="Greys")
    plt.show()
