include("train.jl")

Model() = Chain(
    Dense(256 => 128, swish),
    LSTM(128 => 512),
    Dense(512 => 256, swish)
)

const tc = moveto(OnGPU, StochasticDescent(32, Encodings), load(TrainState, "model", Model, ADAM()))
