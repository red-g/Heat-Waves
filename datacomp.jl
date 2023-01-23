using BSON: @load, @save
using Flux
using Statistics: mean
using StatsBase: sample

function heatscorerows()
    @load "hwscores.bson" hwscores
    hwscores |> eachrow
end

const rows = heatscorerows() |> gpu
const data = Flux.DataLoader(rows; batchsize=32, shuffle=true)

const fullsize = 1000
const encodingsize = 128

const auto = Chain(
    enc=Chain(Dense(fullsize => 512, swish), Dense(512 => 256, swish), Dense(256 => encodingsize, swish)),
    dec=Chain(Dense(encodingsize => 256, swish), Dense(256 => 512, swish), Dense(512 => fullsize, σ))
) |> gpu
const mparams = Flux.params(auto)

loss(y) = Flux.mse(auto(y), y)
mloss(ys) = loss.(ys) |> mloss

sampleindexable(x, n) = x[sample(1:length(x), n)]

testloss() = sampleindexable(rows, length(rows) ÷ 10) |> mloss

const opt = ADAM()

function train!()
    for ys in data
        g = Flux.gradient(() -> mloss(ys), mparams)
        Flux.update!(opt, mparams, g)
    end
end

for e in 1:100
    train!()
    @info "Epoch $e" loss = testloss()
end

@save "autoencoder.bson" auto

function encode()
    encodings = Matrix{Float32}(undef, length(rows), encodingsize)
    for (i, r) in enumerate(rows)
        encodings[i, :] = auto[:enc](r)
    end
    @save "encodings.bson" encodings
end