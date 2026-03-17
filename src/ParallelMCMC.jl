module ParallelMCMC

using AbstractMCMC
using MCMCChains
using LinearAlgebra
using Random
using Statistics

include("MALA/MALA.jl")
include("DEER/DEER.jl")
include("interface.jl")

export DensityModel, BatchedDensityModel
export MALASampler, MALATransition, MALAState
export AdaptiveMALASampler, AdaptiveMALATransition, AdaptiveMALAState
export DEERSampler, DEERTransition, DEERState, MALATapeElement
export MALA, DEER

end
