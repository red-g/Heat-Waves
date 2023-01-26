using BSON: @load
using StatsBase: sample

#Wraps a matrix, returning row views like eachrow, but with indexing
struct RowsIter{M<:AbstractMatrix}
    matrix::M
end
Base.getindex(r::RowsIter, i::Int) = @view r.matrix[i, :]
Base.getindex(r::RowsIter, i::Union{AbstractArray,Colon}) = RowsIter(@view r.matrix[i, :])
Base.length(r::RowsIter) = size(r.matrix, 1)
Base.size(r::RowsIter) = (length(r),)
Base.lastindex(r::RowsIter) = length(r)
Base.iterate(r::RowsIter, i=1) = i > length(r) ? nothing : (r[i], i + 1)
Flux.gpu(r::RowsIter) = RowsIter(Flux.gpu(r.matrix))

#loads randomized batches, ordered so that context is continuous
struct SequentialLoader{D,B}
    data::D
    batchsize::B
end

function indicesToRanges(indices)#could use preallocated array
    ranges = []
    i₋₁ = 1
    for i in indices
        push!(ranges, i₋₁:(i-1))
        i₋₁ = i
    end
    ranges
end

#indexes start at 2 because the network always needs something to predict off of
function Base.iterate(s::SequentialLoader, state=nothing)
    indices = sample(2:length(s.data), s.batchsize, ordered=true, replace=false)
    ranges = indicesToRanges(indices)
    x = Iterators.map(r -> s.data[r], ranges)
    y = s.data[indices]
    (zip(x, y), state)
end

Base.IteratorSize(::Type{<:SequentialLoader}) = Base.IsInfinite()

function proportionScores(trainPortion=0.9)
    @load "hwscores.bson" HeatScores
    scores = RowsIter(HeatScores)
    cutoff = round(Int, length(scores) * trainPortion)
    train = scores[1:cutoff]
    test = scores[(cutoff+1):end]
    train, test
end

function proportionEncodings(trainPortion=0.9)
    @load "encodings.bson" encodings
    scores = RowsIter(encodings)
    cutoff = round(Int, length(scores) * trainPortion)
    train = scores[1:cutoff]
    test = scores[(cutoff+1):end]
    train, test
end
#try loading 39 years, one for testing. 
#this would make the test error less accurate but reflects real world scenarios better
#there is no need to use a 4 year old model

scoresToPredictions(scores, days=1) = zip(scores, Iterators.drop(scores, days))