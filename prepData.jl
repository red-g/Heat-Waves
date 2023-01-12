using BSON: @save
using CSV
using CSV.Dates
using Flux
using Statistics: mean
using JET

const Data = CSV.File("daily_temperature_1000_cities_1980_2020.csv")
const Columns::Vector{Vector{Union{String,Missing}}} = getfield.(Data.columns, :column)
const TempStartRow = 14
const Cols = length(Columns)
const LabelsCol = 1
const DataStartCol = LabelsCol + 1
const CitiesRow = 2
const Rows = length(Columns[1])
const TempRows = Rows - TempStartRow + 1
const TempCols = Cols - DataStartCol + 1

function dateTempMatrixFrom(cols)
    temps = Matrix{Float32}(undef, TempRows, TempCols)
    for (i, col) in enumerate(cols[DataStartCol:end])
        temps[:, i] = parse.(Float32, col[TempStartRow:end])
    end
    temps
end

const QuadYear = (4 * 365) + 1
const DistFromQYCenter = (QuadYear - 1) ÷ 2
const QYCenter = DistFromQYCenter + 1

function meanYearTemps(temps)
    meanTemps = Matrix{Float32}(undef, QuadYear, TempCols)
    for day in 1:QuadYear
        meanTemps[day, :] = mean(temps[day:QuadYear:end, :], dims=1)
    end
    meanTemps
end

function calcResiduals(temps, meanTemps)
    residuals = similar(temps)
    for day in 1:TempRows
        quadDay = ((day - 1) % QuadYear) + 1
        residuals[day, :] = temps[day, :] - meanTemps[quadDay, :]
    end
    residuals
end

function heatScores(temps, meanTemps)
    scores = Matrix{Float32}(undef, TempRows, TempCols)
    residuals = calcResiduals(temps, meanTemps)
    for day in 1:TempRows
        weights = day:-1:1
        residualsₜ = residuals[1:day, :] |> eachcol
        scores[day, :] = residualsₜ .|> (col -> col ./ weights) .|> sum .|> σ
    end#sigmoid may be a mistake: it factors in both hot and cold waves.
    scores
end#to only measurement heatwaves: cap every score < 0, sub the mean

const Temps = dateTempMatrixFrom(Columns)
const MeanTemps = meanYearTemps(Temps)
const HeatScores = heatScores(Temps, MeanTemps)

@save "hwscores.bson" HeatScores

function weightedmean(data, point)
    weights = 1:length(data) .|> p -> 1 / (abs(point - p) + 1)
    sum(data .* weights) / sum(weights)
end

#the idea behind a year mean is to create a less volatile daily mean, taking into account the other days around the target day
function yearmeans(temps)
    ymeans = Matrix{Float32}(undef, TempRows, TempCols)
    for day in 1:TempRows
        qs = max(1, day - DistFromQYCenter)
        qe = min(TempRows, days + DistFromQYCenter)
        qtemps = temps[qs:qe, :]
        ymeans[day, :] = weightedmean.(eachcol(qtemps), QYCenter)
    end
    ymeans
end#this should work fine, but the mean temps at the final and starting years could be a little off