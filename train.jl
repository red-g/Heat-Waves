using Flux
import Flux: setup
using BSON: @save
import Flux.Optimise: AbstractOptimiser
using Accessors

include("data.jl")
include("loss.jl")

abstract type RunLocation end
struct OnGPU <: RunLocation end
struct OnCPU <: RunLocation end

moveto(::Type{OnGPU}, v) = gpu(v)
moveto(::Type{OnCPU}, v) = cpu(v)

abstract type DescentMode end
struct StochasticDescent{I<:Integer} <: DescentMode
    batchsize::I
    test::E
    train::A
end
StochasticDescent(batchsize) = StochasticDescent(batchsize, proportionScores()...)
moveto(rl, dm::StochasticDescent) = StochasticDescent(dm.batchsize, moveto(rl, dm.test), moveto(rl, dm.train))
loss(::StochasticDescent, m, b) = loss(m, b)
data(s::StochasticDescent) = SequentialLoader(s.train, s.batchsize)
testloss(::StochasticDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))

struct SmoothDescent <: DescentMode
    test::E
    train::A
end
SmoothDescent() = SmoothDescent(proportionScores()...)
moveto(rl, dm::SmoothDescent) = SmoothDescent(moveto(rl, dm.test), moveto(rl, dm.train))
loss(::SmoothDescent, m, b) = onedayloss(m, b)
data(s::SmoothDescent) = Iterators.repeated(scoresToPredictions(s.train))
testloss(::SmoothDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))

struct TrainState{P,M,O}
    path::P
    model::M
    opt::O
end

Flux.setup(ts::TrainState) = setup(ts.opt, ts.model)
moveto(rl, ts::TrainState) = TrainState(ts.path, moveto(rl, ts.model), ts.opt)

function save(ts::TrainState)
    (; model, opt) = moveto(OnCPU, ts)
    @save ts.path model opt
end

function load(::Type{TrainState}, path, defM, defopt)
    fullpath = "$(path).bson"
    if isfile(fullpath)
        println("Loading train state from $(fullpath)")
        @load fullpath model opt
        TrainState(fullpath, model, opt)
    else
        println("Initializing train state")
        TrainState(fullpath, defM(), defopt)
    end
end

struct TrainConfig{D<:DescentMode,S<:TrainState}
    descmode::D#how the model will learn——does not change
    state::S#the actual information of the model & opt——changes
end

moveto(rl, tc::TrainConfig) = TrainConfig(moveto(rl, tc.descmode), moveto(rl, tc.state))

function train!((; descmode, state)::TrainConfig, updates)
    println("Initial test loss: $(testloss(descmode, state.model))")
    optstate = setup(state)
    for (i, b) in enumerate(Iterators.take(data(descmode), updates))
        Flux.reset!(state.model)
        (grad,) = Flux.gradient(m -> loss(descmode, m, b), state.model)
        Flux.update!(optstate, state.model, grad)
        println("Batch $(i). Test loss: $(testloss(descmode, state.model))")
    end
    save(state)
end