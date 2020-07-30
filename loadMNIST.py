import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random
import numpy as np


def loadMNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train/255
    x_test = x_test/255
    x_train = [sample.reshape(1, 784)[0] for sample in x_train]
    x_test = [sample.reshape(1, 784)[0] for sample in x_test]
    y_train = [add_one(y) for y in y_train]
    y_test = [add_one(y) for y in y_test]
    print("mnist loaded")
    return (x_train, y_train), (x_test, y_test)

def add_one(index):
    arr = np.zeros(10)
    arr[index] = 1.0
    return arr

def plotMNIST_random():
    index = random.randint(0, 59999)
    (x_train, y_train), (x_test, y_test) = loadMNIST()
    print("Image number {index}. It is a {number}".format(index=index, number=y_train[index]))
    plt.imshow(x_train[index], cmap="Greys")
    plt.show()


def plotMNIST(index):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    plt.imshow(x_test[index], cmap="Greys")
    plt.show()
