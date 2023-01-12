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

function heatScore(temps, meanTemps)
    scores = Matrix{Float32}(undef, TempRows, TempCols)
    residuals = calcResiduals(temps, meanTemps)
    for day in 1:TempRows
        weights = day:-1:1
        residualsₜ = residuals[1:day, :] |> eachcol
        scores[day, :] = residualsₜ .|> (col -> col ./ weights) .|> sum .|> σ
    end
    scores
end

const Temps = dateTempMatrixFrom(Columns)
const MeanTemps = meanYearTemps(Temps)
const HeatScores = heatScore(Temps, MeanTemps)

@save "hwscores.bson" HeatScores

#weighted mean

function weightedmean(data, point)
    weights = 1:length(data) .|> p -> 1 / (abs(point - p) + 1)
    sum(data .* weights) / sum(weights)
end#create dims options; look at how it is done in julia docs

function weightedmean(data, point, dims)
end

function yearmeans(temps)
    ymeans = Matrix{Float32}(undef, TempRows, TempCols)
    qyStart = 1
    qyEnd = QuadYear
    for day in 1:TempRows#make more efficient by iterating by year, not day
        if day > qyEnd
            qyStart = qyEnd + 1
            qyEnd = qyStart + QuadYear - 1
        end
        qyOffset = day - qyStart
        qyTemps = temps[qyStart:qyEnd, :]
        ymeans[day, :] = weightedmean.(eachcol(qyTemps), qyOffset)
    end
    ymeans
end

struct Every{N}
    n::N
end

Base.iterate(e::Every, state=0) = (state ÷ e.n, state + 1)

Base.IteratorSize(::Type{<:Every}) = Base.IsInfinite()

function yearmeans(temps)
    ymeans = Matrix{Float32}(undef, TempRows, TempCols)
    for (qy, day) in zip(Every(QuadYear), 1:TempRows)
        qs = (qy * QuadYear) + 1#the length of the list is not a perfect multiple
        qe = qs + QuadYear
        qtemps = temps[qs:qe, :]
        qoffset = day - qs
        ymeans[day, :] = weightedmean.(eachcol(qtemps), qoffset)
    end
end