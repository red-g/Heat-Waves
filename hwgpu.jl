include("train.jl")

const coredata = CoreData(ModelInfo("model"), OnGPU)

function save(m)
    model = cpu(m)
    @save "model.bson" model
end

train!(coredata, SmoothDescent(100, Descent(0.01)), save)