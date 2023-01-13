using Flux
using Statistics: mean

#For stochastic descent——batches
function loss(m, x, y)
    local ŷ
    for xᵢ in x
        ŷ = m(xᵢ)
    end
    Flux.mse(ŷ, y)
end

loss(m, d) = sum(dᵢ -> loss(m, dᵢ...), d) / length(d)

#For gradient descent——prediction pairs
onedayloss(m, x, y) = Flux.mse(m(x), y)

onedayloss(m, d) = sum(dᵢ -> onedayloss(m, dᵢ...), d) / length(d)

function testloss(m, context, preds)
    Flux.reset!(m)
    m.(context)
    preds .|> (d -> onedayloss(m, d...)) |> mean
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
