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

#=const Data = CSV.File("GlobalLandTemperaturesByCity.csv")

const CityNames = unique(Data.City)
const CityNum = length(CityNames)
const CityNameDict = zip(CityNames, 1:CityNum) |> Dict{String,Int}

OneHotData = Tuple{Vector{Flux.OneHotVector},Vector{Float32}}
VectorTuple{A,B} = Tuple{Vector{A},Vector{B}}
CityHWData = VectorTuple{Vector{Vector{Float32}},VectorTuple{Flux.OneHotVector,Float32}}
#gives input onehot list
cityNameToOneHot(city) = Flux.onehot(city, CityNames)
cityNumToOneHot(city) = Flux.onehot(city, 1:CityNum)
#the new dataset is arranged horizontally
struct CityTemp
    date::Dates.Date
    city::String
    temp::Union{Float32,Missing}
end

mapMissing(f) = x -> ismissing(x) ? missing : f(x)

const cityTemps = CityTemp.(
    Data.dt,
    String.(Data.City),
    mapMissing(Float32).(Data.AverageTemperature),
)

const years = Data.dt .|> Dates.year |> unique
const maxYear = maximum(years)
const minYear = minimum(years)
const yearDiff = maxYear - minYear + 1
const maxMonthDays = 31
const monthsInYear = 12
#test alternative method using dictionary since so many are missing
function sortCityTemps(cts)
    scts = Array{Union{Float32,Missing}}(missing, maxMonthDays, monthsInYear, yearDiff, CityNum)
    for ct in cts
        day = Dates.day(ct.date)
        month = Dates.month(ct.date)
        year = Dates.year(ct.date) - minYear + 1
        scts[day, month, year, CityNameDict[ct.city]] = ct.temp
    end
    scts
end



meanCityTemps(scts) = eachslice(scts; dims=3) |> zipSplat .|> skipmissing .|> meanOrMissing

meanOrMissing(x) = isempty(x) ? mean(x) : missing

function tempsTohwScores!(scts, mcts)
    residuals = eachslice(scts; dims=3) .- Ref(mcts)
    indices = scts |> skipmissing |> eachindex
    for (p, i) in enumerate(indices)
        residualsᵢ = getindex(residuals, indices[1:p])
        timeWeightsᵢ = indices[p:-1:1]
        scts[i] = (residualsᵢ ./ timeWeightsᵢ) |> sum |> σ
    end
    scts
end

function hwsToNetInputs(hws)
    function hwsToOneHots(i, cityHws)
        onehot = cityNumToOneHot(i)
        mapMissing(chw -> cat(onehot, chw)).(cityHws)
    end
    toDayMatrices(cityOneHots) = cityOneHots |> zipSplat .|> skipmissing .|> hcats
    eachslice(hws; dims=4) |> enumerate .|> Base.splat(hwsToOneHots) |> toDayMatrices |> removeEmpties
end

removeEmpties(x) = filter(!isempty, x)

tovec(ar) = reshape(ar, :)

zipSplat(cs) = ZipSplat(cs)

mutable struct ZipSplat{C}
    cs::C
end

Base.iterate(c::ZipSplat, state=1) =
    state > length(c.cs) ?
    nothing : (getindex.(c.cs, state), state + 1)

Base.length(c::ZipSplat) = length(c.cs[1])

function hcats(vectors::Vector{Vector{T}}) where {T}
    hcatted = Matrix{T}(undef, length(vectors[1]), length(vectors))
    for (i, v) in enumerate(vectors)
        hcatted[:, i] = v
    end
    hcatted
end

function createHeatWaveScores(cts)
    scts = sortCityTemps(cts)
    println("Sorted datastructure: $(typeof(scts))")
    mcts = meanCityTemps(scts)
    println("Mean datastructure: $(typeof(mcts))")
    chws = tempsTohwScores!(scts, mcts)
    println("Heatwave datastructure: $(typeof(chws))")
    hwsToNetInputs(chws)
end

#const heatWaveScores = createHeatWaveScores(cityTemps)
#println("Final datastructure: $(typeof(heatWaveScores))")
#@save "heatWaveScores.bson" heatWaveScores=#