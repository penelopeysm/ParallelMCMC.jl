module MALA

using Random, LinearAlgebra

# Preconditioner dispatch helpers.  cholM is either nothing (identity) or a Cholesky factor.
_apply_M(g, ::Nothing) = g
_apply_M(g, cholM::Cholesky) = cholM.L * (cholM.L' * g)

_apply_L(ξ, ::Nothing) = ξ
_apply_L(ξ, cholM::Cholesky) = cholM.L * ξ

_quad_Minv(r, ::Nothing) = dot(r, r)
function _quad_Minv(r, cholM::Cholesky)
    w = cholM.L \ r
    return dot(w, w)
end

_logdet_M(::Nothing) = 0.0
_logdet_M(cholM::Cholesky) = logdet(cholM)

"""
Compute the log of the MALA proposal density q(y | x).

With mass matrix M (passed as `cholM = cholesky(M)`):
  y ~ Normal(x + ϵ M ∇logp(x), 2ϵ M)

With `cholM=nothing` (default), uses identity M = I.
"""
function logq_mala(
    y::AbstractVector, x::AbstractVector, gradlogp_x::AbstractVector, ϵ::Real;
    cholM=nothing,
)
    μ = x .+ ϵ .* _apply_M(gradlogp_x, cholM)
    d = length(x)
    r = y .- μ
    return -0.5 * _quad_Minv(r, cholM) / (2ϵ) - (d / 2) * log(4π * ϵ) - 0.5 * _logdet_M(cholM)
end

"""
One taped MALA step.

Inputs:
- logp(x): log density (up to constant)
- gradlogp(x): gradient of log density
- x: current state
- ϵ: step size
- ξ: N(0, I) noise vector (tape)
- u: Uniform(0,1) scalar (tape)
- cholM: optional Cholesky factor of mass matrix M (default: identity)

Returns:
- x_next
"""
function mala_step_taped(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))
    g_x = gradlogp(x)

    y = x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)

    logp_x = logp(x)
    logp_y = logp(y)

    g_y = gradlogp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)

    logα = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

    return (log(u) < logα) ? y : x
end

"""
Run sequential taped MALA for T steps.

Inputs:
- x0: initial state
- ξs: vector of ξ_t noise vectors, length T
- us: vector of u_t uniforms, length T

Returns:
- xs: Vector of states length T+1, xs[1]=x0, xs[t+1]=x_t
"""
function run_mala_sequential_taped(
    logp,
    gradlogp,
    x0::AbstractVector,
    ϵ::Real,
    ξs::Vector{<:AbstractVector},
    us::AbstractVector;
    cholM=nothing,
)
    T = length(us)
    length(ξs) == T || throw(DimensionMismatch("ξs and us must have the same length"))
    xs = Vector{typeof(x0)}(undef, T + 1)
    xs[1] = copy(x0)
    x = copy(x0)
    for t in 1:T
        x = mala_step_taped(logp, gradlogp, x, ϵ, ξs[t], us[t]; cholM=cholM)
        xs[t + 1] = copy(x)
    end
    return xs
end

"Compute the MALA proposal map y = x + ϵ M ∇logp(x) + √(2ϵ) L ξ."
function mala_proposal(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector; cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    g_x = gradlogp(x)
    return x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)
end

"Compute log acceptance ratio logα(x→y) for MALA."
function mala_logα(
    logp, gradlogp, x::AbstractVector, y::AbstractVector, ϵ::Real; cholM=nothing,
)
    g_x = gradlogp(x)
    g_y = gradlogp(y)
    logp_x = logp(x)
    logp_y = logp(y)
    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)
    return (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)
end

"""
Primal accept indicator for a taped MALA step.
Returns Float64 in {0.0, 1.0} so it can be used as a constant gate.
"""
function mala_accept_indicator(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    y = mala_proposal(logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    logα = mala_logα(logp, gradlogp, x, y, ϵ; cholM=cholM)
    return (log(u) < logα) ? 1.0 : 0.0
end

"""
One taped MALA step, returning both the next state and the accept flag.

This is the efficient entry point: it evaluates `gradlogp` exactly twice (once at `x`,
once at the proposal `y`), compared to calling `mala_accept_indicator` + `mala_step_taped`
separately which evaluates it five times.

Returns `(x_next, accepted::Bool)`.
"""
function mala_step_full(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, u::Real;
    cholM=nothing,
)
    length(x) == length(ξ) || throw(DimensionMismatch("x and ξ must have the same length"))
    0.0 < u < 1.0 || throw(ArgumentError("u must be in (0, 1)"))

    g_x = gradlogp(x)
    y = x .+ ϵ .* _apply_M(g_x, cholM) .+ sqrt(2ϵ) .* _apply_L(ξ, cholM)

    logp_x = logp(x)
    logp_y = logp(y)
    g_y = gradlogp(y)

    logq_y_given_x = logq_mala(y, x, g_x, ϵ; cholM=cholM)
    logq_x_given_y = logq_mala(x, y, g_y, ϵ; cholM=cholM)
    logα = (logp_y + logq_x_given_y) - (logp_x + logq_y_given_x)

    accepted = log(u) < logα
    x_next = accepted ? y : x
    return x_next, accepted
end

"""
Stop-gradient surrogate step used for Jacobians.
`a` (0.0 or 1.0) must be provided as a constant by the DEER machinery.
"""
function mala_step_surrogate(
    logp, gradlogp, x::AbstractVector, ϵ::Real, ξ::AbstractVector, a::Real;
    cholM=nothing,
)
    y = mala_proposal(logp, gradlogp, x, ϵ, ξ; cholM=cholM)
    return (a .* y) .+ ((1 - a) .* x)
end

end # module
