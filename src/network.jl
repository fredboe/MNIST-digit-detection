# declare every variable
using Random

mutable struct Network
    layers::Array{Int64}
    num_layers::Int64
    weights::Array{Matrix{Float64}}
    biases::Array{Vector{Float64}}
end

function initialize(layers::Array{Int64})# nn::Network
    num_layers = length(layers)
    weights = [randn((a, b)) for (a, b) in zip(layers[1:end], layers[0:end])]
    biases = [randn(size_l) for size_l in layers[1:end]]
    return Network(layers, num_layers, weights, biases)
end

function sigmoid(z::Vector{Float64})
    return 1.0./(map(x->exp(-x), z).+1.0)
end

function sigmoid_derivative(z::Vector{Float64})
    expo = map(x->exp(x), z)
    return expo./((expo.+1.0).^2)
end

function cost_derivative(out_a, y)
    return out_a-y
end

function feedforward(a::Vector{Float64}, nn::Network)
    for (w,b) in zip(nn.weights, nn.biases)
        a = sigmoid(w*a+b)
    end
    return a
end


function SGD(train_data::Array{Tuple{Vector{Float64}, Vector{Float64}}}, test_data::Array{Tuple{Vector{Float64}, Vector{Float64}}}, learning_rate=3.0::Float64, epochs=10::Int64, mini_size=30::Int64, nn::Network)
    for epoch in 1:epochs
        println("Epoch: $epoch")
        train_data = shuffle(train_data)
        for i in 1:mini_size:length(train_data) #maybe wrong => probably the last one is not going to have the correct size
            mini_batch = train_data[i:i+mini_size-1] #maybe wrong
            updateWB(mini_batch, learning_rate, mini_size, nn)
        end
        #println(performance)
    end
end

function updateWB(mini_batch, lrate, mini_size, nn)
    change_w = [zeros(size(w)) for w in nn.weights]
    change_b = [zeros(size(b)) for b in nn.biases]
    for sample in mini_batch
        change_w_delta, change_b_delta = backprop(sample[0], sample[1], nn)
        change_w = [w1+w2 for (w1, w2) in zip(change_w, change_w_delta)]
        change_b = [b1+b1 for (b1, b2) in zip(change_b, change_b_delta)]
    end
    nn.weights = [w-(lrate/mini_size)*wc for (w, wc) in zip(nn.weights, change_w)]
    nn.biases = = [b-(lrate/mini_size)*bc for b, bc in zip(nn.biases, change_b)]
end

function backprop(a, y, nn)
    change_w = [zeros(size(w)) for w in nn.weights]
    change_b = [zeros(size(b)) for b in nn.biases]
    activations = [a_in]
    zs = []
    for (w, b) in zip(nn.weights, nn.biases)
        z = w*a+b
        push!(z, zs)
        a=sigmoid(z)
        push!(a, activations)
    end

    error = cost_derivative(activations[end], y)
    change_b[end] = error
    change_w[end] = error.*transpose(a[end-1])

    for l in 1:nn.num_layers-2
        error = (transpose(nn.weights[end-l+1])*error).*sigmoid_derivative(activations[end-l])
        change_b[end-l] = error
        change_w[end-l] = error.*transpose(a[end-l-1])
    end

    return (change_w, change_b)

end


function main()
    nn = initialize([784, 30, 10])
    println("Process completed")
end

main()
