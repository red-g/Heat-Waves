using Flux
using BSON: @save
using Accessors

include("data.jl")
include("loss.jl")

struct CoreData{M,D,T}
    model::M
    data::D
    test::T
end

testloss(cd::CoreData) = testloss(cd.model, cd.data, cd.test)

abstract type HyperParameters end

struct StochasticDescent{B,E,O} <: HyperParameters
    batchsize::B
    updates::E
    opt::O
end

struct SmoothDescent{E,O} <: HyperParameters
    updates::E
    opt::O
end

function train!(lf, cd, opt, sv)
    println("Initial test loss: $(testloss(cd))")
    optstate = Flux.setup(opt, cd.model)
    for (i, b) in enumerate(cd.data)
        (grad,) = Flux.gradient(m -> lf(m, b), cd.model)
        Flux.update!(optstate, cd.model, grad)
        println("Batch $(i). Test loss: $(testloss(cd))")
    end
    sv(cd.model)
end

#this is slower than traditional gradient descent, though it may produce better results when training does complete
function train!(cd::CoreData, sd::StochasticDescent, sv)
    d = Iterators.take(SequentialLoader(cd.data, sd.batchsize), sd.updates)
    train!(loss, (@set cd.data = d), sd.opt, sv)
end
#train!(coredata, StochasticDescent(512, 1000, Descent(0.01)), save)

#while this is usually the slower, more memory intensive method, in this case it is actually the most efficient
function train!(cd::CoreData, sd::SmoothDescent, sv)
    d = Iterators.repeated(scoresToPredictions(cd.data), sd.updates)
    train!(onedayloss, (@set cd.data = d), sd.opt, sv)
end
#gradientdescent(coredata, SmoothDescent(100, Descent(0.01)), save)