# MNIST-digit-detection

## Project
This project contains a basic implementation of a neural network. The goal of the network is to predict the MNIST dataset.

## Execute the code
To see the result of the code is just execute network.py.
```
  python network.py
```
Furthermore, you can try to play around with some parameters. For example, you can change the mini batch size, the learning rate, the epochs or the layout of the network.
```
net = Network([784, 30, 10])     <- line 168
```
```
def SGD(self, train_data, test_data, learning_rate=3.0, epochs=10, mini_size=30):     <- line 77
```
```
net.SGD(train_data, test_data)     <- line 172
```
In addition, it is possible to implement other activation functions, a different weight and bias initializations or other cost functions.

## Further reading
I really recommend you to read the book of Michael Nielsen. (IT'S FREE)
* http://neuralnetworksanddeeplearning.com/chap1.html
