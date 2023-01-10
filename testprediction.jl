using BSON: @load
using Statistics: mean

include("model.jl")

@load "model.bson" model

include("data.jl")

const training, testing = proportionScores(0.9)

function (m::Model)(h, x)
    inp = m.inp(x)
    h′, ŷ = m.rnn.cell(h, inp)
    h′, m.out(ŷ)
end

constatePredict(x, d) = foldl((p, _) -> model(p...), 1:d, init=(state(model), x))[2]

longTermLoss(x, y, days) = Flux.mse(constatePredict(model(x), days - 1), y)

#number does not match up with that of the training measurement; performance may be much better than currently measured
function mlossForDays(context, data, days)
    model.(context)
    preds = scoresToPredictions(data, days)
    l = preds .|> (pred -> longTermLoss(pred..., days)) |> mean
    reset!(model)
    l
end


#mlossForDays(training, testing, 1)