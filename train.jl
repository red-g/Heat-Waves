using Flux
using BSON: @save
import Flux.Optimise: AbstractOptimiser
using Accessors

include("data.jl")
include("loss.jl")
include("model.jl")

abstract type RunLocation end
struct OnGPU <: RunLocation end
struct OnCPU <: RunLocation end

abstract type DescentMode end
struct StochasticDescent{I<:Integer} <: DescentMode
    batchsize::I
end
struct SmoothDescent <: DescentMode end

struct TrainParams::{L, T, D}
    loss::L
    testloss::T
    dataloader::D
end

function TrainParams(::SmoothDescent, training, testing)
    loss = onedayloss
    preds = scoresToPredictions(testing)
    testloss = m -> testloss(m, training, preds)
    dataloader = Iterators.repeated(scoresToPredictions(training))
    TrainParams(loss, testloss, dataloader)
end

function TrainParams(s::StochasticDescent, training, testing)
    loss = loss
    preds = scoresToPredictions(testing)
    testloss = m -> testloss(m, training, preds)
    dataloader = SequentialLoader(training, s.batchsize)
    TrainParams(loss, testloss, dataloader)
end

struct TrainState{L<:RunLocation,M,O,P}
    model::M
    optstate::O
    path::P
end
TrainState(m, o::AbstractOptimiser, p, r) = TrainState{r}(m, Flux.setup(o, m), "$(p).bson")

save(path, model, optstate) = @save path model optstate

save(ts::TrainState{OnCPU}) = save(ts.path, ts.model, ts.opstate)

save(ts::TrainState{OnGPU}) = save(ts.path, cpu(ts.model), ts.opstate)

function load(::Type{TrainState}, path, opt)
    fullpath = "$(path).bson"
    if isfile(fullpath)
        @load ts.path model optstate
        TrainState(model, opstate, path)
    else
        TrainState(Model(), opt, path)
    end
end

load(::Type{TrainState{OnCPU}}, path, opt) = load(TrainState, path, opt)

function load(::Type{TrainState{OnGPU}}, path, opt)
    cts = load(TrainState, path, opt)
    @set cts.model = gpu(cts.model)
end

struct TrainConfig{P<:TrainParams,S<:TrainState}
    params::P
    state::S
end

proportionScores(::Type{OnCPU}, args...) = proportionScores(args...)
proportionScores(::Type{OnGPU}, args...) = gpu.(proportionScores(args...))

function TrainConfig(dm, opt, path, c::Type{<:RunLocation})
    training, testing = proportionScores(c)
    params = TrainParams(dm, training, testing)
    state = load(TrainState{c}, path, opt)
    TrainConfig(params, state)
end

function train!((; params, state)::TrainConfig, updates)
    println("Initial test loss: $(params.testloss(state.model))")
    for (i, b) in enumerate(Iterators.take(params.dataloader, updates))
        Flux.reset!(state.model)
        (grad,) = Flux.gradient(m -> params.loss(m, b), state.model)
        Flux.update!(state.optstate, state.model, grad)
        println("Batch $(i). Test loss: $(params.testloss(model))")
    end
    save(state)
end

# const tc = TrainConfig(StochasticDescent(512), Descent(0.01), "trainstate", OnCPU)
# train!(tc, 1000)