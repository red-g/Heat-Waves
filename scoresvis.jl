using Plots
using BSON: @load

@load "hwscoresrev.bson" HeatScores

historicCityTemps(c) = HeatScores[:, c]
tempSnapShot(t) = HeatScores[t, :]

plotHistoricCityTemps(c) = plot(historicCityTemps(c), title="Historic Heat Scores for City #$c")

plotTempSnapShot(t) = plot(tempSnapShot(t), title="Heat Scores on day #$t")