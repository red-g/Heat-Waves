using Flux
import Flux: setup
using BSON: @save
import Flux.Optimise: AbstractOptimiser

include("data.jl")
include("loss.jl")

abstract type RunLocation end
struct OnGPU <: RunLocation end
struct OnCPU <: RunLocation end

moveto(rl, v) = _moveto(rl, v)
moveto(rl, v...) = moveto.(rl, v)
_moveto(::Type{OnGPU}, v) = gpu(v)
_moveto(::Type{OnCPU}, v) = cpu(v)

abstract type DataSet end
struct Scores <: DataSet end
struct Encodings <: DataSet end

abstract type DescentMode end

# Normally the more performant option, as it uses batch updates; in this case it is actually slower than SmoothDescent 
# since predictions are sequential. It may still yield better results, however
struct StochasticDescent{I<:Integer,E,A} <: DescentMode
    batchsize::I
    train::E
    test::A
end
StochasticDescent(batchsize, ::Type{Scores}) = StochasticDescent(batchsize, proportionScores()...)
StochasticDescent(batchsize, ::Type{Encodings}) = StochasticDescent(batchsize, proportionEncodings()...)
moveto(rl, dm::StochasticDescent) = StochasticDescent(dm.batchsize, moveto(rl, dm.train), moveto(rl, dm.test))
loss(::StochasticDescent, m, b) = loss(m, b)
data(s::StochasticDescent) = SequentialLoader(s.train, s.batchsize)
testloss(s::StochasticDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))
Base.show(io::IO, s::StochasticDescent) = print(io, "StochasticDescent($(s.batchsize))")
lossinterval(::StochasticDescent) = (test=15, train=75)

# Uses all samples each step; more performant than normal because predictions are sequential: 
# it is only marginally more work to compute the gradients of all instead of some
struct SmoothDescent{E,A} <: DescentMode
    train::E
    test::A
end
SmoothDescent(::Type{Scores}) = SmoothDescent(proportionScores()...)
SmoothDescent(::Type{Encodings}) = SmoothDescent(proportionEncodings()...)
moveto(rl, dm::SmoothDescent) = SmoothDescent(moveto(rl, dm.train), moveto(rl, dm.test))
loss(::SmoothDescent, m, b) = onedayloss(m, b)
data(s::SmoothDescent) = Iterators.repeated(scoresToPredictions(s.train))
testloss(s::SmoothDescent, m) = testloss(m, s.train, scoresToPredictions(s.test))
Base.show(io::IO, ::SmoothDescent) = print(io, "SmoothDescent")
lossinterval(::SmoothDescent) = (test=1, train=5)

struct TrainState{P,M,O}
    path::P
    model::M
    opt::O
end

Flux.setup(ts::TrainState) = setup(ts.opt, ts.model)
moveto(rl, ts::TrainState) = TrainState(ts.path, moveto(rl, ts.model), ts.opt)
Base.show(io::IO, ::TrainState) = print(io, "TrainState")

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
    intervals = lossinterval(dm)
    println("Initial test loss: $(testloss(dm, ts.model))")
    optstate = setup(ts)
    for (i, b) in enumerate(Iterators.take(data(dm), updates))
        Flux.reset!(ts.model)
        (grad,) = Flux.gradient(m -> loss(dm, m, b), ts.model)
        Flux.update!(optstate, ts.model, grad)
        i % intervals.test == 0 && println("Batch $(i).\n\tTest loss: $(testloss(dm, ts.model))")
        i % intervals.train == 0 && println("\tLoss: $(loss(dm, ts.model, b))")
    end
    save(ts)
end