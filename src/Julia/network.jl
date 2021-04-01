module NN

using Random

sigmoid(z :: Vector{Float64}) = @. 1.0/(1.0+exp(-z))
sigmoid_der(z :: Vector{Float64}) = @. exp(z)/(1.0+exp(z))^2
cost(a :: Vector{Float64}, y :: Vector{Float64}) = @. 0.5*(a-y)^2
cost_der(a :: Vector{Float64}, y :: Vector{Float64}) = @. (a-y)


mutable struct Network
    layers :: Vector{Int}
    num_layers :: Int
    weights :: Vector{Matrix{Float64}}
    biases :: Vector{Vector{Float64}}

    zs :: Vector{Vector{Float64}}
    as :: Vector{Vector{Float64}}

    change_w1 :: Vector{Matrix{Float64}}
    change_b1 :: Vector{Vector{Float64}}
    change_w2 :: Vector{Matrix{Float64}}
    change_b2 :: Vector{Vector{Float64}}
end

function init(layers)
    num_layers = length(layers)
    weights = [randn(i,j)./sqrt(2) for (i,j) in zip(layers[2:end], layers[1:end-1])]
    biases = [randn(i)./sqrt(2) for i in layers[2:end]]
    zs = [zeros(i) for i in layers[2:end]]
    as = [zeros(i) for i in layers]
    change_w1 = [zeros(i,j) for (i,j) in zip(layers[2:end], layers[1:end-1])]
    change_b1 = [zeros(i) for i in layers[2:end]]
    change_w2 = [zeros(i,j) for (i,j) in zip(layers[2:end], layers[1:end-1])]
    change_b2 = [zeros(i) for i in layers[2:end]]
    return Network(layers, num_layers, weights, biases, zs, as,
                            change_w1, change_b1, change_w2, change_b2)
end

function feedforward!(net)
    for i=2:net.num_layers
        net.zs[i-1] .= net.weights[i-1]*net.as[i-1] .+ net.biases[i-1]
        net.as[i] .= sigmoid(net.zs[i-1])
    end
end

function predict(net, a_in)
    net.as[1] .= a_in
    feedforward!(net)
    return argmax(net.as[end])
end

function performance(net, test_data)
    return sum([predict(net,sample) == argmax(label) for (sample,label) in test_data]) / length(test_data)
end


function SGD!(net, train_data, test_data, lrate=3.0, epochs=5, mini_size=30)
    for epoch=1:epochs
        println("Epoch: $epoch")
        Random.shuffle(train_data)
        @time begin
        for i=1:mini_size:length(train_data)
            updateWB!(net, train_data[i:i+mini_size-1], lrate)
        end
        end
        perform = performance(net, test_data)
        println("Performance at Epoch $epoch is $perform")
    end
end


function updateWB!(net, mini_batch, lrate)
    len = length(mini_batch)
    constant = lrate / len
    for sample in mini_batch
        backprop!(net, sample[1], sample[2])
        net.change_w1 .+= net.change_w2
        net.change_b1 .+= net.change_b2
    end
    net.change_w1 .*= constant
    net.change_b1 .*= constant
    net.weights .-= net.change_w1
    net.biases .-= net.change_b1
end

function outer_error!(net, y)
    #error = cost_der(net.as[end], y)
    net.change_b2[end] .= cost_der(net.as[end], y)
    net.change_w2[end] .= net.change_b2[end] .* transpose(net.as[end-1])
end

function inner_error!(net)
    #error = (transpose(net.weights[end-l+1])*error).*sigmoid_der(net.zs[end-l])
    for l=1:net.num_layers-2
        net.change_b2[end-l] .= (transpose(net.weights[end-l+1])*net.change_b2[end-l+1]).*sigmoid_der(net.zs[end-l])
        net.change_w2[end-l] .= net.change_b2[end-l] .* transpose(net.as[end-l-1])
    end
end

function backprop!(net, a, y)
    net.as[1] .= a
    feedforward!(net)
    outer_error!(net, y)
    inner_error!(net)
end

function isZero(mat)
    x=true
    for i=1:length(mat)
        x = mat[i] == zeros(size(mat[i]))
    end
    return x
end

end  # module NN
