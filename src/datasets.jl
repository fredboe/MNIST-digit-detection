module datasets

using PyCall

const pickle = pyimport("pickle")


function loadMNIST()
    println("Collecting the MNIST data...")
    x_train = pickle.load(open("MNIST/x_train.pkl"))
    y_train = pickle.load(open("MNIST/y_train.pkl"))
    x_test = pickle.load(open("MNIST/x_test.pkl"))
    y_test = pickle.load(open("MNIST/y_test.pkl"))
    train_data = collect(zip(x_train,y_train))
    test_data = collect(zip(x_test, y_test))
    println("Loading MNIST: done")
    return train_data, test_data
end

end #module datasets
