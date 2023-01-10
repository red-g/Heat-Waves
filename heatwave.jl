using Flux
using BSON: @save

include("data.jl")
include("model.jl")
include("loss.jl")

const hwpred = tryload("model")

const training, testing = proportionScores(0.9)
const testingpreds = scoresToPredictions(testing)

function train!(lf, model, d, opt)
    println("Initial test loss: $(testloss(model, training, testingpreds))")
    optstate = Flux.setup(opt, model)
    for (i, b) in enumerate(d)
        (grad,) = Flux.gradient(m -> lf(m, b), model)
        Flux.update!(optstate, model, grad)
        tl = testloss(model, training, testingpreds)
        println("Batch $(i). Test loss: $(tl)")
    end
    @save "model.bson" model
end

#this is slower than traditional gradient descent, though it may produce better results when training does complete
function stochasticdescent(m, bs, e, opt)
    d = SequentialLoader(training, bs)
    train!(loss, m, Iterators.take(d, e), opt)
end
#stochasticdescent(hwpred, 512, 1000, Descent(0.01))

#while this is usually the slower, more memory intensive method, in this case it is actually the most efficient
function gradientdescent(m, e, opt)
    d = Iterators.repeated(scoresToPredictions(training), e)
    train!(onedayloss, m, d, opt)
end
#gradientdescent(hwpred, 100, Descent(0.01))

#use more sophisticated optimizer, like ADAM