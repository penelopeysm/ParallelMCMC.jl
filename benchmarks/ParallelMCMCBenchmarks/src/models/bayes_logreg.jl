module BayesLogReg

using Random
using LinearAlgebra

"""
    make_data(rng, N, D; β_scale=1.0) -> (X, y, β_true)

Generate synthetic Bayesian logistic regression data.
"""
function make_data(rng::AbstractRNG, N::Int, D::Int; β_scale::Real=1.0)
    β_true = randn(rng, D) .* float(β_scale)
    X = randn(rng, N, D)
    logits = X * β_true
    y = Float64[rand(rng) < 1 / (1 + exp(-l)) ? 1.0 : 0.0 for l in logits]
    return X, y, β_true
end

"""
    make_problem(X, y) -> (logp, gradlogp)

Optimized to reuse memory buffers for GPU performance.
"""
function make_problem(X::AbstractMatrix, y::AbstractVector)
    # Pre-allocate workspaces in the closure scope
    N = size(X, 1)
    logits = similar(X, N)
    p = similar(X, N)
    resid = similar(X, N)

    function logp(β::AbstractVector)
        mul!(logits, X, β)
        # ll = sum(@. y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits))))
        # We perform this in-place to avoid allocations
        # Note: log1p(exp(x)) is the softplus function
        @. p = y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits)))
        return sum(p) - 0.5 * sum(abs2, β)
    end

    function gradlogp(β::AbstractVector)
        mul!(logits, X, β)
        @. p = 1 / (1 + exp(-logits))
        @. resid = y - p
        # grad = X' * (y - p) - β
        # We reuse the output allocation here by using mul!
        grad = (X' * resid) .- β
        return grad
    end

    return logp, gradlogp
end

"""
    make_problem_with_hvp(X, y) -> (logp, gradlogp, hvp)
"""
function make_problem_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp, gradlogp = make_problem(X, y)

    # Workspace buffers
    N = size(X, 1)
    logits = similar(X, N)
    p = similar(X, N)
    w = similar(X, N)
    Xv = similar(X, N)

    function hvp(β::AbstractVector, v::AbstractVector)
        mul!(logits, X, β)
        @. p = 1 / (1 + exp(-logits))
        @. w = p * (1 - p)
        mul!(Xv, X, v)
        @. Xv = w * Xv
        return -(X' * Xv) .- v
    end

    return logp, gradlogp, hvp
end

"""
    make_problem_batched(X, y) -> (logp_batch, gradlogp_batch)
"""
function make_problem_batched(X::AbstractMatrix, y::AbstractVector)
    # Pre-allocate workspaces for batch size M
    # Note: We determine M from input size in the functions

    function logp_batch(B::AbstractMatrix)
        N, M = size(X, 1), size(B, 2)
        logits = similar(X, N, M)
        mul!(logits, X, B)

        # In-place calculation for log-posterior
        # Using specific buffers avoids the massive allocations in the original
        res = similar(logits)
        @. res = y * (-log1p(exp(-logits))) + (1 - y) * (-log1p(exp(logits)))

        # Sum over rows for the likelihood, subtract quadratic regularization
        return vec(sum(res; dims=1)) .- 0.5 .* vec(sum(abs2, B; dims=1))
    end

    function gradlogp_batch(B::AbstractMatrix)
        N, M = size(X, 1), size(B, 2)
        logits = similar(X, N, M)
        mul!(logits, X, B)

        @. logits = 1 / (1 + exp(-logits)) # Reuse logits buffer as p
        resid = similar(logits)
        @. resid = y .- logits

        return (X' * resid) .- B
    end

    return logp_batch, gradlogp_batch
end

"""
    make_problem_batched_with_hvp(X, y) -> (logp_batch, gradlogp_batch, hvp_batch)
"""
function make_problem_batched_with_hvp(X::AbstractMatrix, y::AbstractVector)
    logp_batch, gradlogp_batch = make_problem_batched(X, y)

    function hvp_batch(B::AbstractMatrix, V::AbstractMatrix)
        size(B) == size(V) || throw(DimensionMismatch("B and V must have the same size"))
        N, M = size(X, 1), size(B, 2)

        # Reuse logic with pre-allocated buffers
        logits = similar(X, N, M)
        mul!(logits, X, B)

        @. logits = 1 / (1 + exp(-logits)) # logits is now p
        w = similar(logits)
        @. w = logits * (1 - logits)

        Xv = similar(logits)
        mul!(Xv, X, V)
        @. Xv = w * Xv

        return -(X' * Xv) .- V
    end

    return logp_batch, gradlogp_batch, hvp_batch
end

end # module
