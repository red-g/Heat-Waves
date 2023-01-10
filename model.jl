using Flux
using BSON: load

Model() = Chain(Dense(1000 => 256, swish), LSTM(256 => 512), Dense(512 => 1000, σ))

function tryload(path::String, name=Symbol(path))
    fullpath = "$(path).bson"
    if isfile(fullpath)
        load(fullpath)[name]
    else
        Model()
    end
end