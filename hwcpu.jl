include("train.jl")
include("model.jl")

const hwpred = tryload("model")
const training, testing = proportionScores(0.9)
const testpreds = scoresToPredictions(testing)
const coredata = CoreData(hwpred, training, testpreds)

save(model) = @save "model.bson" model

train!(coredata, SmoothDescent(100, Descent(0.01)), save)