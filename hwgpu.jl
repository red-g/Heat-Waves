include("train.jl")

Model() = Chain(
    :pre=Dense(256 => 128, swish),
    :mem=LSTM(128 => 512),
    :post=Dense(512 => 256, swish)
)

const tc = moveto(OnGPU, StochasticDescent(32, Encodings), load(TrainState, "model", Model, ADAM()))
