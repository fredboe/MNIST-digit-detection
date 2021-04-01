include("datasets.jl")
include("network.jl")

using .datasets
using .NN


function main()
    train_data, test_data = datasets.loadMNIST()
    net = NN.init([784, 30, 10])
    println("Initialize Network: done")
    NN.SGD!(net, train_data, test_data)
    println("Program stoped")
end

main()
