include("train.jl")

Model() = Chain(Dense(CityNum => 256, swish), LSTM(256 => 512), Dense(512 => CityNum, σ))

const tc = SmoothDescent(Scores), load(TrainState, "trainstate", Model, Descent(0.01))

train!(tc, 100)