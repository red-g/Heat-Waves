using Flux
using Statistics: mean

#For stochastic descent——batches
function loss(m, x, y)
    local ŷ
    for xᵢ in x
        ŷ = m(xᵢ)
    end
    Flux.mse(ŷ, y)
end

loss(m, d) = sum(dᵢ -> loss(m, dᵢ...), d) / length(d)

#For gradient descent——prediction pairs
onedayloss(m, x, y) = Flux.mse(m(x), y)

onedayloss(m, d) = sum(dᵢ -> onedayloss(m, dᵢ...), d) / length(d)

function testloss(m, context, preds)
    Flux.reset!(m)
    Flux.testmode!(m)
    m.(context)#in case of gpu, this may legimately be parallel
    l = preds .|> (d -> onedayloss(m, d...)) |> mean
    Flux.testmode!(m, false)
    l
end

function testlossSeq(m, context, preds)
    Flux.reset!(m)
    Flux.testmode!(m)
    for c in context
        m(c)
    end
    l = foldl(preds; init=0.0f0) do s, p
        s + onedayloss(m, p...)
    end / length(preds)#that or sum / length
    Flux.testmode!(m, false)
    l
end

ExpandedModel(a, m) = Chain(a[:enc], m, a[:dec])

function Auto()
    @load "autoencoder.bson" model
    model
end

function expandedloss(m)
    auto = Auto() |> gpu
    em = ExpandedModel(auto, m)
    train, test = proportionScores() .|> gpu
    testlossSeq(em, train, scoresToPredictions(test))
end

#checks if both the sample and batch gradient are functional
function testgradient(m, lf, d)
    b = first(d)
    s = first(d)
    sg = Flux.gradient(m -> lf(m, s...), m)
    println("Sample Gradient: $(sg)")
    bg = Flux.gradient(m -> lf(m, b), m)
    println("Batch Gradient: $(bg)")
    nothing
end

#calculate average difference within encodings and full representation
#that is to say, find the mean dif between one day snapshots for encodings and full representation