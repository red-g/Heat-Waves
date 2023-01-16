using Plots
using CSV
using SparseArrays

#broken by new changes
#change background color from white

include("testprediction.jl")

#dual display showing predictions for test data d days in advance, as compared to actual data
const testingPreds = scoresToPredictions(testing)

const d = 1

const Data = CSV.File("daily_temperature_1000_cities_1980_2020.csv")
const Columns = getfield.(Data.columns, :column)
const Cities = getindex.(Columns[2:end], 2)
const Latitudes = getindex.(Columns[2:end], 3)
const Longitudes = getindex.(Columns[2:end], 4)

function coordsToInds(x)
    floats = parse.(Float32, x)
    ints = round.(Int, floats)
    ints .- minimum(ints) .+ 1
end

const LatInds = coordsToInds(Latitudes)
const LngInds = coordsToInds(Longitudes)

scorestoHeatmap(hs) = sparse(LatInds, LngInds, 100 * hs .|> σ) |> collect |> heatmap

model.(training)

const actual = @animate for (i, y) ∈ enumerate(testing)[(1+d):end]
    hm = scorestoHeatmap(y)
    plot(hm, title="Day $(i)", zmin=0, zmax=1, xaxis=false, yaxis=false)
end

gif(actual, "actual$(d).gif", fps=10)

const predicted = @animate for (i, x) ∈ enumerate(testing[1:(end-d)])
    ŷ = constatePredict(model(x), d - 1)
    hm = scorestoHeatmap(ŷ)
    plot(hm, title="Day $(i + d)", zmin=0, zmax=1, xaxis=false, yaxis=false)
end

gif(predicted, "predicted$(d).gif", fps=10)
