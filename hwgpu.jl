include("train.jl")
include("model.jl")

const hwpred = tryload("model") |> gpu
const training, testing = proportionScores(0.9) .|> gpu
const testpreds = scoresToPredictions(testing)
const coredata = CoreData(hwpred, training, testpreds)

function save(m)
    model = cpu(m)
    @save "model.bson" model
end

train!(coredata, SmoothDescent(100, Descent(0.01)), save)