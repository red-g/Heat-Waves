include("train.jl")

Model() = Chain(Dense(1000 => 256, swish), LSTM(256 => 512), Dense(512 => 1000, σ))

const tc = SmoothDescent(), load(TrainState, "trainstate", Model, Descent(0.01))

train!(tc, 100)