import random
import numpy as np


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    return np.exp(z)/((np.exp(z)+1)**2)


class Network:

    def __init__(self, layers):
        self.layers = layers
        self.weights = [np.random.randn(a, b) for a, b in zip(layers[1:], layers[0:])]
        self.biases = [np.random.randn(layer) for layer in layers[1:]]

    def initialize(self):
        pass

    def feedforward(self, a):
        for w, b in zip(self.weights, self.biases):
            print(np.dot(w,a)+b)
            a = sigmoid(np.dot(w, a) + b)
        return a

    def all_as_zs(self, a_in):
        activation = [a_in]
        a = a_in
        Z = []
        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, a) + b
            a = sigmoid(z)
            Z.append(z)
            activation.append(a)
        return activation, Z

    def predict(self, a_in):
        a_out = self.feedforward(a_in)
        print(a_out)
        return np.argmax(a_out)

    def SGD(self, training_data, learning_rate=3.0, epochs=3, mini_size=32):
        for epoch in range(0, epochs):
            print("Epoch: {epoch}".format(epoch=epoch))
            random.shuffle(training_data)
            for i in range(0, len(training_data), mini_size):
                mini_batch = training_data[i:i+mini_size]
                self.updateWB(mini_batch, learning_rate)

    def updateWB(self, mini_batch, lrate):
        length = len(mini_batch)
        change_w = [np.zeros(a, b) for a, b in zip(self.layers[1:], self.layers[0:])]
        change_b = [np.zeros(layer) for layer in self.layers[1:]]
        for sample in mini_batch:
            change_w_delta, change_b_delta = self.backprop(sample[1], sample[0])
            change_w = [w1+w2 for w1, w2 in zip(change_w, change_w_delta)]
            change_b = [b1+b2 for b1, b2 in zip(change_b, change_b_delta)]
        self.weights = [w-lrate/length*wc for w, wc in zip(self.weights, change_w)]
        self.biases = [b-lrate/length*bc for b, bc in zip(self.biases, change_b)]

    def backprop(self, y, a_in):
        weight_change = []
        bias_change = []
        # feedforward
        a, z = self.all_as_zs(a_in)
        # output error
        error = np.multiply(self.cost_derivative(y, a[-1]), sigmoid_derivative(z[-1]))
        bias_change.append(error)
        weight_change.append(np.multiply(error[None].T, a[-1]))
        # backpropagate the error
        for i in range(2, len(self.layers)):
            weight = np.transpose(self.weights[-i+1])
            error = np.multiply(np.dot(weight, error), sigmoid_derivative(z[-i]))
            # output: gradient -> multiply with a
            bias_change.append(error)
            weight_change.append(np.multiply(error[None].T, a[-i-1]))
        return weight_change[::-1], bias_change[::-1]

    def cost_function(self, y, a_out):
        return 0.5*(y-a_out)**2

    def cost_derivative(self, y, a_out):
        return (y-a_out)


if __name__ == "__main__":
    n = Network((3, 3, 2))
    print(n.predict(np.array([0.5, 0.5, 0.5])))
    n.backprop(np.array([0.5,0.5]), np.array([0.5,0.5,0.5]))
