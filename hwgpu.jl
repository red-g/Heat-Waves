include("train.jl")

Model() = Chain(Dense(RowSize => 512, swish), LSTM(512 => 2048), Dense(2048 => 1024, swish), Dense(1024 => RowSize, σ))

const dm = SmoothDescent()
const ts = load(TrainState, "trainstate", Model, ADAM())
const tc = moveto(OnGPU, TrainConfig(dm, ts))

train!(tc, 100)