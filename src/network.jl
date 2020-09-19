
module NeuralNetwork
using Random
using DelimitedFiles


function loadMNIST()
    println("collecting the data...")
    x_train = readdlm("data/x_train.txt")
    x_train2 = [x_train[row,:] for row in 1:size(x_train)[1]]
    y_train = readdlm("data/y_train.txt")
    y_train2 = [y_train[row,:] for row in 1:size(y_train)[1]]
    x_test = readdlm("data/x_test.txt")
    x_test2 = [x_test[row,:] for row in 1:size(x_test)[1]]
    y_test = readdlm("data/y_test.txt")
    y_test2 = [y_test[row,:] for row in 1:size(y_test)[1]]
    train_data = []
    test_data = []
    for (x, y) in zip(x_train2, y_train2)
        push!(train_data, (x, y))
    end
    for (x, y) in zip(x_test2, y_test2)
        push!(test_data, (x, y))
    end
    println("data loading completed")
    return train_data, test_data
end



mutable struct Network
    layers::Array{Int64}
    num_layers::Int64
    weights::Array{Matrix{Float64}}
    biases::Array{Vector{Float64}}
end

function initialize(layers::Array{Int64}) :: Network
    num_layers :: Int64 = length(layers)
    weights :: Array{Matrix{Float64}} = [randn((a, b))./sqrt(2) for (a, b) in zip(layers[2:end], layers[1:end])]
    biases :: Array{Vector{Float64}} = [randn(size_l)./sqrt(2) for size_l in layers[2:end]]
    return Network(layers, num_layers, weights, biases)
end

function sigmoid(z::Vector{Float64}) :: Vector{Float64}
    return 1.0./(1.0.+MathConstants.e.^(-z))
end

function sigmoid_derivative(z::Vector{Float64}) :: Vector{Float64}
    return MathConstants.e.^(z)./((1.0.+MathConstants.e.^(z)).^2)
end

function cost_derivative(out_a :: Vector{Float64}, y :: Vector{Float64}) :: Vector{Float64}
    return out_a-y
end

function feedforward!(a::Vector{Float64}, nn::Network) :: Vector{Float64}
    for (w,b) in zip(nn.weights, nn.biases)
        a = sigmoid(w*a+b)
    end
    return a
end

function SGD(train_data, test_data, learning_rate::Float64, epochs::Int, mini_size::Int, nn::Network)

    for epoch in 1:epochs
        println("Epoch: $epoch")
        train_data = shuffle(train_data)
        @time begin
        for i in 1:mini_size:length(train_data)
                mini_batch = train_data[i:i+mini_size-1]
                updateWB(mini_batch, learning_rate, mini_size, nn)
        end
        end
        println("learning finished")
        performance :: Float64 = evaluate(test_data, nn)
        println("Performance at epoch $epoch is $performance")
    end
end

function updateWB(mini_batch, lrate::Float64, mini_size::Int, nn::Network)
    change_w = [zeros(size(w)) for w in nn.weights]
    change_b = [zeros(size(b)) for b in nn.biases]
    for sample in mini_batch
        change_w_delta, change_b_delta = backprop(sample[1], sample[2], nn)
        change_w = [w1+w2 for (w1, w2) in zip(change_w, change_w_delta)]
        change_b = [b1+b1 for (b1, b2) in zip(change_b, change_b_delta)]
    end

    nn.weights = [w-(lrate/mini_size)*wc for (w, wc) in zip(nn.weights, change_w)]
    nn.biases = [b-(lrate/mini_size)*bc for (b, bc) in zip(nn.biases, change_b)]
end

function backprop(a::Vector{Float64}, y::Vector{Float64}, nn::Network)
    change_w :: Array{Matrix{Float64}} = [zeros(size(w)) for w in nn.weights]
    change_b :: Array{Vector{Float64}} = [zeros(size(b)) for b in nn.biases]
    activations = [a]
    zs = []
    for (w, b) in zip(nn.weights, nn.biases)
        z :: Vector{Float64}= w*a+b
        push!(zs, z)
        a=sigmoid(z)
        push!(activations, a)
    end

    # error :: Vector{Float64} = cost_derivative(activations[end], y).*sigmoid_derivative(zs[end])
    error :: Vector{Float64} = cost_derivative(activations[end], y)
    change_b[end] = error
    change_w[end] = error.*transpose(activations[end-1])

    for l in 1:nn.num_layers-2
        error = (transpose(nn.weights[end-l+1])*error).*sigmoid_derivative(zs[end-l])
        change_b[end-l] = error
        change_w[end-l] = error.*transpose(activations[end-l-1])
    end

    return (change_w, change_b)
end

function predict(a_in::Vector{Float64}, nn::Network) :: Int
    a_out :: Vector{Float64} = feedforward!(a_in, nn)
    return argmax(a_out)-1
end

function evaluate(test_data, nn::Network) :: Float64
    len :: Int = length(test_data)
    return sum([predict(a, nn)==(argmax(l)-1) for (a, l) in test_data])/len
end

end

function main()
    train_data, test_data = NeuralNetwork.loadMNIST()
    nn = NeuralNetwork.initialize([784, 30, 10])
    NeuralNetwork.SGD(train_data, test_data, 3.0, 10, 30, nn)
    println("Process completed")
end

main()
