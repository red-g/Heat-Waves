using BSON: @load
using Statistics: mean

#broken by new changes

@load "model.bson" model

include("data.jl")

const training, testing = proportionScores()

function (m::Model)(h, x)
    inp = m.inp(x)
    h′, ŷ = m.rnn.cell(h, inp)
    h′, m.out(ŷ)
end

constatePredict(m, x, d) = foldl((p, _) -> m(p...), 1:d, init=(state(m), x))[2]

longTermLoss(m, x, y, days) = Flux.mse(constatePredict(m, m(x), days - 1), y)

function mlossForDays(m, context, data, days)
    m.(context)
    preds = scoresToPredictions(data, days)
    l = preds .|> (pred -> longTermLoss(pred..., days)) |> mean
    reset!(m)
    l
end


#mlossForDays(model, training, testing, 1)