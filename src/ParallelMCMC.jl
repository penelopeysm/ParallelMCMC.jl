module ParallelMCMC

using AbstractMCMC
using CUDA
using Enzyme
using MCMCChains
using LinearAlgebra
using Random
using Statistics

include("MALA/MALA.jl")
include("DEER/DEERScan.jl")
include("DEER/DEER.jl")
include("interface.jl")

export DensityModel
export MALASampler, MALATransition, MALAState
export AdaptiveMALASampler, AdaptiveMALATransition, AdaptiveMALAState
export ParallelMALASampler, ParallelMALATransition, ParallelMALAState
export solve_affine_seq!, solve_affine_seq, solve_affine_scan_diag!, solve_affine_scan_diag,
       check_affine_scan, affine_scan_residual
export MALA, DEER

end
