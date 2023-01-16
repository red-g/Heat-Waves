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

moveto(rl, v) = _moveto(rl, v)
moveto(rl, v...) = moveto.(rl, v)
_moveto(::Type{OnGPU}, v) = gpu(v)
_moveto(::Type{OnCPU}, v) = cpu(v)

abstract type DescentMode end
struct StochasticDescent{I<:Integer,E,A} <: DescentMode
    batchsize::I
    test::E
    train::A
end
StochasticDescent(batchsize) = StochasticDescent(batchsize, proportionScores()...)
moveto(rl, dm::StochasticDescent) = StochasticDescent(dm.batchsize, moveto(rl, dm.test), moveto(rl, dm.train))
loss(::StochasticDescent, m, b) = loss(m, b)
data(s::StochasticDescent) = SequentialLoader(s.train, s.batchsize)
testloss(::StochasticDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))

struct SmoothDescent{E,A} <: DescentMode
    test::E
    train::A
end
SmoothDescent() = SmoothDescent(proportionScores()...)
moveto(rl, dm::SmoothDescent) = SmoothDescent(moveto(rl, dm.test), moveto(rl, dm.train))
loss(::SmoothDescent, m, b) = onedayloss(m, b)
data(s::SmoothDescent) = Iterators.repeated(scoresToPredictions(s.train))
testloss(s::SmoothDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))

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

function train!((dm, ts), updates)
    println("Initial test loss: $(testloss(dm, ts.model))")
    optstate = setup(ts)
    for (i, b) in enumerate(Iterators.take(data(dm), updates))
        Flux.reset!(ts.model)
        (grad,) = Flux.gradient(m -> loss(dm, m, b), ts.model)
        Flux.update!(optstate, ts.model, grad)
        println("Batch $(i). Test loss: $(testloss(dm, ts.model))")
    end
    save(ts)
end