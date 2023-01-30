using BSON: @save
using CSV
using CSV.Dates
using Flux
using Statistics: mean

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

function weightedmean(data, point)
    weights = 1:length(data) .|> p -> 1 / ((abs(point - p) + 1) .^ 2)
    sum(data .* weights) / sum(weights)
end
#to prevent reliance on future times, the soft temps could also be calculated just looking at the past QY of temperatures

function weightedmean(data)
    weights = 1 ./ ((length(data):-1:1) .^ 2)#added to weigh heavily against far away temperatures
    sum(data .* weights) / sum(weights)
end#this places too much emphasis on far away temperatures; leads to cyclical data

#the idea behind a year mean is to create a less volatile daily mean, taking into account the other days around the target day
function softTemps(temps)
    ymeans = similar(temps)
    midstart, midfinish = QuadYear, TempRows - DistFromQYCenter
    for day in 1:midstart
        qe = day + DistFromQYCenter
        qtemps = eachcol(temps[begin:qe, :])
        ymeans[day, :] = weightedmean.(qtemps, day)
    end
    for day in (midstart+1):midfinish
        qs = day - DistFromQYCenter
        qe = day + DistFromQYCenter
        qtemps = eachcol(temps[qs:qe, :])
        ymeans[day, :] = weightedmean.(qtemps, QYCenter)
    end
    for day in (midfinish+1):TempRows
        qs = day - DistFromQYCenter
        qtemps = eachcol(temps[qs:end, :])
        ymeans[day, :] = weightedmean.(qtemps, QYCenter)
    end
    ymeans
end

#alternate version of softtemps that only looks at past temperatures, not future ones
function pastSoftTemps(temps)
    ymeans = similar(temps)
    for day in 1:QuadYear
        qtemps = eachcol(temps[begin:day, :])
        ymeans[day, :] = weightedmean.(qtemps)
    end
    for day in (QuadYear+1):TempRows
        qtemps = eachcol(temps[(day-QuadYear):day, :])
        ymeans[day, :] = weightedmean.(qtemps)
    end
    ymeans
end

function calcResiduals(temps, meanTemps)
    residuals = similar(temps)
    for day in 1:TempRows
        quadDay = ((day - 1) % QuadYear) + 1
        residuals[day, :] = temps[day, :] - meanTemps[quadDay, :]
    end
    residuals
end

discount(cap, day) = log(1 + 1.0f0 / (cap - day + 1))

function heatScores(temps, meanTemps)
    scores = similar(temps)
    residuals = calcResiduals(temps, meanTemps)
    for day in 1:TempRows
        weights = discount.(day, 1:day)
        residualsₜ = residuals[1:day, :] |> eachcol
        scores[day, :] = residualsₜ .|> (col -> col .* weights) .|> sum
    end
    scores
end
#sigmoid may be a mistake: it factors in both hot and cold waves.
#to only measurement heatwaves: cap every score < 0, sub the mean
#zscore could be best solution. 
#Though we may want to do a different one for each city, since they represent different, albeit related, distributions.
#Also perhaps update dataset to a global map, taken from satellite/other estimates.
#Cities leave out a lot space where not many people, even though that space still has a big impact on the global climate.

function createHeatScores()
    temps = dateTempMatrixFrom(Columns)
    meantemps = (meanYearTemps ∘ softTemps)(temps)
    HeatScores = heatScores(temps, meantemps)
    @save "hwscores.bson" HeatScores
end