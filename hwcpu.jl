include("train.jl")

const coredata = CoreData(ModelInfo("model"), OnCPU)

save(model) = @save "model.bson" model

train!(coredata, SmoothDescent(100, Descent(0.01)), save)