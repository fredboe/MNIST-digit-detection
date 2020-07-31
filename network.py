import random
import numpy as np
import loadMNIST


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
        return np.argmax(a_out)

    def SGD(self, train_data, test_data, learning_rate=3.0, epochs=10, mini_size=30):
        for epoch in range(1, epochs+1):
            print("Epoch: {epoch}".format(epoch=epoch))
            random.shuffle(train_data)
            for i in range(0, len(train_data), mini_size):
                mini_batch = train_data[i:i+mini_size]
                self.updateWB(mini_batch, learning_rate)
            print("Performance at epoch {epoch}: {perf}".format(epoch=epoch, perf=self.performance(test_data)))

    def updateWB(self, mini_batch, lrate):
        length = len(mini_batch)
        change_w = [np.zeros(w.shape) for w in self.weights]
        change_b = [np.zeros(b.shape) for b in self.biases]
        for sample in mini_batch:
            change_w_delta, change_b_delta = self.backprop(sample[1], sample[0])
            change_w = [w1+w2 for w1, w2 in zip(change_w, change_w_delta)]
            change_b = [b1+b2 for b1, b2 in zip(change_b, change_b_delta)]
        self.weights = [w-(lrate/length)*wc for w, wc in zip(self.weights, change_w)]
        self.biases = [b-(lrate/length)*bc for b, bc in zip(self.biases, change_b)]

    def backprop(self, y, a_in):
        change_w = []
        change_b = []
        # feedforward
        a, z = self.all_as_zs(a_in)
        # output error
        error = np.multiply(self.cost_derivative(a[-1], y), sigmoid_derivative(z[-1]))
        change_b.append(error)
        change_w.append(np.multiply(error[None].T, a[-2]))
        # backpropagate the error
        for i in range(2, len(self.layers)):
            weight = np.transpose(self.weights[-i+1])
            error = np.multiply(np.dot(weight, error), sigmoid_derivative(z[-i]))
            # output: gradient -> multiply with a
            change_b.append(error)
            change_w.append(np.multiply(error[None].T, a[-i-1]))
        return change_w[::-1], change_b[::-1]

    def cost_function(self, a_out, y):
        return 0.5*(a_out-y)**2

    def cost_derivative(self, a_out, y):
        return (a_out-y)

    def performance(self, test_data):
        return sum([net.predict(d) == np.argmax(l) for d, l in test_data])/len(x_test)


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = loadMNIST.loadMNIST()
    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))
    net = Network([784, 30, 10])
    print(net.performance(test_data))
    net.SGD(train_data, test_data, epochs=10)
    print("Resulted performance: {perf}".format(perf=net.performance(test_data)))
