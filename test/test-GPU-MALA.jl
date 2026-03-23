using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
const MALA = ParallelMCMC.MALA

import CUDA

# Check if a real GPU is accessible by attempting a small allocation.
# CUDA.functional() only checks that the library loads, not that a device exists.
const CUDA_AVAILABLE = try
    CUDA.CuArray([1f0])
    true
catch
    false
end

if !CUDA_AVAILABLE
    @info "No CUDA GPU detected — skipping GPU tests."
else

# Standard normal target: logp(X) = -0.5 ||X||², grad = -X
# Broadcasting and sum work on CuArrays, so these are GPU-compatible.
logp_batch(X)     = vec(-0.5f0 .* sum(abs2, X; dims=1))
gradlogp_batch(X) = -X

@testset "GPU inputs stay on device" begin
    D, N = 4, 32

    X = CUDA.randn(Float32, D, N)
    Ξ = CUDA.randn(Float32, D, N)
    u = CUDA.rand(Float32, N)

    X_next, accepted = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u)

    @test X_next isa CUDA.CuArray
    @test accepted isa CUDA.CuArray
    @test size(X_next)  == (D, N)
    @test length(accepted) == N
end

@testset "GPU and CPU produce identical results (same seed)" begin
    D, N = 3, 16

    # Generate random numbers on CPU, copy to GPU.
    rng = MersenneTwister(42)
    X_cpu = randn(rng, Float32, D, N)
    Ξ_cpu = randn(rng, Float32, D, N)
    u_cpu = rand(rng, Float32, N)

    X_gpu = CUDA.CuArray(X_cpu)
    Ξ_gpu = CUDA.CuArray(Ξ_cpu)
    u_gpu = CUDA.CuArray(u_cpu)

    X_next_cpu, acc_cpu = MALA.mala_step_batched(logp_batch, gradlogp_batch, X_cpu, 0.1f0, Ξ_cpu, u_cpu)
    X_next_gpu, acc_gpu = MALA.mala_step_batched(logp_batch, gradlogp_batch, X_gpu, 0.1f0, Ξ_gpu, u_gpu)

    @test Array(X_next_gpu) ≈ X_next_cpu   atol=1f-5
    @test Array(acc_gpu)    == acc_cpu
end

@testset "GPU stationary distribution (standard normal)" begin
    D, N, T = 3, 512, 1_000

    X = CUDA.randn(Float32, D, N)

    for _ in 1:T
        Ξ = CUDA.randn(Float32, D, N)
        u = CUDA.rand(Float32, N)
        X, _ = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1f0, Ξ, u)
    end

    X_cpu = Array(X)
    mu   = vec(mean(X_cpu; dims=2))
    vars = vec(var(X_cpu;  dims=2))

    @test maximum(abs.(mu))          < 0.15
    @test maximum(abs.(vars .- 1f0)) < 0.25
end

@testset "GPU DimensionMismatch errors" begin
    X     = CUDA.randn(Float32, 3, 5)
    Ξ_bad = CUDA.randn(Float32, 3, 4)   # wrong N
    u_ok  = CUDA.rand(Float32, 5)
    u_bad = CUDA.rand(Float32, 4)        # wrong N

    @test_throws DimensionMismatch MALA.mala_step_batched(
        logp_batch, gradlogp_batch, X, 0.1f0, Ξ_bad, u_ok,
    )
    @test_throws DimensionMismatch MALA.mala_step_batched(
        logp_batch, gradlogp_batch, X, 0.1f0, CUDA.randn(Float32, 3, 5), u_bad,
    )
end

end # if CUDA_AVAILABLE
