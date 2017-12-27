using MultipleScattering
using JLD


train = collect(values(load("bunny.mst.jld")))[1]
test = collect(values(load("bunnytest.mst.jld")))[1]
