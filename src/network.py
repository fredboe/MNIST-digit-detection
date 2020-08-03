import random
import numpy as np
import loadMNIST


def sigmoid(z):
    """
    returns:
            sigmoid result
    """
    return 1.0/(1.0+np.exp(-z))


def sigmoid_derivative(z):
    """
        returns:
                derivative of sigmoid function
    """
    return np.exp(z)/((np.exp(z)+1)**2)


class Network:

    def __init__(self, layers):
        # layers: list -> each element describes the number of neurons in the layer
        self.layers = layers
        # number of layers in the network
        self.len_layers = len(layers)
        # weights: list of matrices, biases: list of vectors
        self.weights, self.biases = self.initialize(self.layers)

    def initialize(self, layers):
        """
            returns:
                    random weights: list, random biases: list
        """
        # /np.sqrt(b)
        weights = [np.random.randn(a, b) for a, b in zip(layers[1:], layers[0:])]
        biases = [np.random.randn(layer) for layer in layers[1:]]
        return weights, biases

    def feedforward(self, a):
        """
            parameter:
                        a: activation_in -> vector
            returns:
                    activation_out
        """
        for w, b in zip(self.weights, self.biases):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def all_as_zs(self, a_in):
        """
            returns:
                    list of all activations and all z's
        """
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
        """
            predicts the label of the input image
            returns:
                    the index of the max output neuron
        """
        a_out = self.feedforward(a_in)
        return np.argmax(a_out)

    def SGD(self, train_data, test_data, learning_rate=3.0, epochs=10, mini_size=30):
        """
            creates mini batches and passes them to updateWB
        """
        for epoch in range(1, epochs+1):
            print("Epoch: {epoch}".format(epoch=epoch))
            random.shuffle(train_data)
            for i in range(0, len(train_data), mini_size):
                mini_batch = train_data[i:i+mini_size]
                self.updateWB(mini_batch, learning_rate)
            print("Performance at epoch {epoch}: {perf}".format(epoch=epoch, perf=self.performance(test_data)))

    def updateWB(self, mini_batch, lrate):
        """
            parameters:
                        mini_batch: list of 30 training samples
                        lreate: learning rate
            updates the weights and biases with one mini batch
        """
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
        """
            parameters:
                        y: label vector
                        a_in: input activation
            executes backpropagation algorithm
            returns:
                    a list of biases and and a list of weights (change)
        """
        change_w = []
        change_b = []
        # feedforward
        a, z = self.all_as_zs(a_in)
        # output error
        # error = np.multiply(self.cost_derivative(a[-1], y, z[-1]), sigmoid_derivative(z[-1]))
        error = self.out_layer_error_cross_entropy(a[-1], y)
        change_b.append(error)
        change_w.append(np.multiply(error[None].T, a[-2]))
        # backpropagate the error
        for i in range(2, self.len_layers):
            weight = np.transpose(self.weights[-i+1])
            error = np.multiply(np.dot(weight, error), sigmoid_derivative(z[-i]))
            # output: gradient -> multiply with a
            change_b.append(error)
            change_w.append(np.multiply(error[None].T, a[-i-1]))
        return change_w[::-1], change_b[::-1]

    def out_layer_error_cross_entropy(self, a_out, y):
        """
            returns:
                    cross_entropy output error
        """
        return (a_out-y)

    def cost_function(self, a_out, y, z):
        """
            qudratic cost function
        """
        return 0.5*(a_out-y)**2

    def cost_derivative(self, a_out, y, z):
        """
            derivative of quadratic cost function
        """
        return (a_out-y)

    def performance(self, test_data):
        """
            parameters:
                        test_data: list of test data images
            returns:
                the sum of correct predicted images divided by the number of test_data samples
        """
        return sum([net.predict(d) == np.argmax(l) for d, l in test_data])/len(test_data)


if __name__ == "__main__":
    # loads the mnist data
    (x_train, y_train), (x_test, y_test) = loadMNIST.loadMNIST()
    train_data = list(zip(x_train, y_train))
    test_data = list(zip(x_test, y_test))
    # creates the network class: 3 layers
    net = Network([784, 30, 10])
    # prints the performance before learning
    print(net.performance(test_data))
    # network learns
    net.SGD(train_data, test_data)
    # prints the performance after learning
    print("Resulted performance: {perf}".format(perf=net.performance(test_data)))
