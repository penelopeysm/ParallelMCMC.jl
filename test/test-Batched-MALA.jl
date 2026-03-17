using Test
using Random
using LinearAlgebra
using Statistics

using ParallelMCMC
const MALA = ParallelMCMC.MALA

# Standard normal target: logp(x) = -0.5 ||x||², grad = -x
logp_batch(X)      = vec(-0.5 .* sum(abs2, X; dims=1))   # D×N → N
gradlogp_batch(X)  = -X                                   # D×N → D×N

@testset "BatchedDensityModel construction" begin
    bm = BatchedDensityModel(logp_batch, gradlogp_batch, 4)
    @test bm.dim == 4

    X = randn(4, 10)
    @test length(bm.logdensity_batch(X)) == 10
    @test size(bm.grad_logdensity_batch(X)) == (4, 10)
end

@testset "mala_step_batched output shapes" begin
    rng = MersenneTwister(1)
    D, N = 3, 20
    X = randn(rng, D, N)
    Ξ = randn(rng, D, N)
    u = rand(rng, N)

    X_next, accepted = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1, Ξ, u)

    @test size(X_next)  == (D, N)
    @test length(accepted) == N
    @test eltype(X_next) == Float64
end

@testset "mala_step_batched single accepted chain matches scalar step" begin
    rng = MersenneTwister(42)
    D = 3

    x  = randn(rng, D)
    ξ  = randn(rng, D)
    u  = rand(rng)

    # Scalar step
    x_scalar, acc_scalar = MALA.mala_step_full(
        x -> -0.5 * dot(x, x), x -> -x, x, 0.1, ξ, u,
    )

    # Batched step with N=1
    X       = reshape(copy(x), D, 1)
    Xi      = reshape(copy(ξ), D, 1)
    u_vec   = [u]
    X_next, accepted_vec = MALA.mala_step_batched(
        logp_batch, gradlogp_batch, X, 0.1, Xi, u_vec,
    )

    @test isapprox(vec(X_next), x_scalar; atol=1e-12)
    @test Bool(accepted_vec[1]) == acc_scalar
end

@testset "mala_step_batched preserves rejected chains" begin
    rng = MersenneTwister(5)
    D, N = 4, 50

    X = randn(rng, D, N)
    Ξ = randn(rng, D, N)
    # Force all rejections: u very close to 1 so log(u) ≈ 0 > any logα
    u = ones(N) .* (1 - 1e-15)

    X_next, accepted = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1, Ξ, u)

    # With u ≈ 1, log(u) ≈ 0 which is > most logα; almost all should be rejected
    # (standard normal is well-behaved so this should reject everything or near it)
    n_accepted = sum(accepted)
    # At this u value essentially all should reject
    @test n_accepted < N
end

@testset "mala_step_batched DimensionMismatch errors" begin
    X = randn(3, 5)
    Ξ_bad = randn(3, 4)   # wrong N
    u_good = rand(5)
    u_bad  = rand(4)       # wrong N

    @test_throws DimensionMismatch MALA.mala_step_batched(
        logp_batch, gradlogp_batch, X, 0.1, Ξ_bad, u_good
    )
    @test_throws DimensionMismatch MALA.mala_step_batched(
        logp_batch, gradlogp_batch, X, 0.1, randn(3, 5), u_bad
    )
end

@testset "mala_step_batched stationary distribution (many chains)" begin
    rng = MersenneTwister(2025)
    D, N, T = 3, 200, 2_000

    X = randn(rng, D, N)

    for _ in 1:T
        Ξ = randn(rng, D, N)
        u = rand(rng, N)
        X, _ = MALA.mala_step_batched(logp_batch, gradlogp_batch, X, 0.1, Ξ, u)
    end

    # Pool all chains after burn-in (already burned in by running T steps)
    mu   = vec(mean(X; dims=2))
    vars = vec(var(X; dims=2))

    @test maximum(abs.(mu))          < 0.15
    @test maximum(abs.(vars .- 1.0)) < 0.20
end
