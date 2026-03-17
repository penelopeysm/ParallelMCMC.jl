#=
AbstractMCMC interface for ParallelMCMC samplers.

Defines model/sampler/state/transition types and implements
`AbstractMCMC.step` so that `sample(model, sampler, N)` works out of the box.
=#

"""
    DensityModel(logdensity, grad_logdensity, dim)

Wraps a log-density function and its gradient for use with ParallelMCMC samplers.

- `logdensity(x::AbstractVector) -> Real`
- `grad_logdensity(x::AbstractVector) -> AbstractVector`
- `dim::Int` — dimensionality of the parameter space
"""
struct DensityModel{F,G} <: AbstractMCMC.AbstractModel
    logdensity::F
    grad_logdensity::G
    dim::Int
end

"""
    MALASampler(epsilon)

Metropolis-Adjusted Langevin Algorithm sampler with step size `epsilon`.
"""
struct MALASampler <: AbstractMCMC.AbstractSampler
    epsilon::Float64
end

struct MALAState{V<:AbstractVector}
    x::V
    logp::Float64
end

struct MALATransition{V<:AbstractVector}
    x::V
    logp::Float64
    accepted::Bool
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MALASampler;
    initial_params=nothing,
    kwargs...,
)
    x = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, model.dim)
    end
    logp_val = model.logdensity(x)
    t = MALATransition(x, logp_val, true)
    s = MALAState(x, logp_val)
    return t, s
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::MALASampler,
    state::MALAState;
    kwargs...,
)
    x = state.x
    ϵ = sampler.epsilon
    D = model.dim

    ξ = randn(rng, D)
    u = rand(rng)

    x_next, accepted = MALA.mala_step_full(
        model.logdensity, model.grad_logdensity, x, ϵ, ξ, u
    )

    logp_val = accepted ? model.logdensity(x_next) : state.logp
    t = MALATransition(x_next, logp_val, accepted)
    s = MALAState(x_next, logp_val)
    return t, s
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:MALATransition},
    model::DensityModel,
    sampler::MALASampler,
    state::MALAState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    N = length(samples)
    D = model.dim

    names = if param_names !== nothing
        param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp, :accepted]

    vals = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 2)

    for i in 1:N
        s = samples[i]
        vals[i, :] .= s.x
        internals[i, 1] = s.logp
        internals[i, 2] = s.accepted ? 1.0 : 0.0
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    MALATapeElement(ξ, u)

One element of the MALA noise tape: a noise vector `ξ ~ N(0,I)` and a uniform
scalar `u ~ Uniform(0,1)`.  Stored with a concrete vector type `V` for type
stability inside `DEER.TapedRecursion`.
"""
struct MALATapeElement{V<:AbstractVector}
    ξ::V
    u::Float64
end

"""
    DEERSampler(epsilon; T, maxiter, tol_abs, tol_rel, jacobian, damping, probes)

DEER-accelerated MALA sampler.

DEER solves for a trajectory of `T` steps in parallel (O(log T) iterations),
then the AbstractMCMC interface returns samples from that trajectory
sequentially.  When the trajectory is exhausted a new tape is drawn and DEER
re-solves starting from the last state.

# Arguments
- `epsilon` — MALA step size.
- `T` — trajectory length per DEER solve (default 64).
- `maxiter` — maximum DEER iterations per solve (default 200).
- `tol_abs`, `tol_rel` — convergence tolerances (default 1e-6, 1e-5).
- `jacobian` — Jacobian mode: `:diag`, `:stoch_diag`, or `:full` (default `:diag`).
- `damping` — DEER damping in (0,1] (default 0.5; helps convergence).
- `probes` — Hutchinson probes for `:stoch_diag` mode (default 1).

# Parallel chains
Both `MALASampler` and `DEERSampler` are compatible with
`AbstractMCMC.sample(model, sampler, MCMCThreads(), N, nchains)`.  Each chain
has its own immutable state and RNG so there is no shared mutable data.
Note that each `DEERSampler` chain internally uses `Base.Threads.@threads`
for the parallel scan, so running many chains via `MCMCThreads()` on a machine
with few threads may over-subscribe the thread pool.
"""
struct DEERSampler <: AbstractMCMC.AbstractSampler
    epsilon::Float64
    T::Int
    maxiter::Int
    tol_abs::Float64
    tol_rel::Float64
    jacobian::Symbol
    damping::Float64
    probes::Int
end

function DEERSampler(
    epsilon::Real;
    T::Int=64,
    maxiter::Int=200,
    tol_abs::Real=1e-6,
    tol_rel::Real=1e-5,
    jacobian::Symbol=:diag,
    damping::Real=0.5,
    probes::Int=1,
)
    return DEERSampler(
        Float64(epsilon), T, maxiter,
        Float64(tol_abs), Float64(tol_rel),
        jacobian, Float64(damping), probes,
    )
end

"""
State for a `DEERSampler` chain.

- `x` — current position (= `trajectory[:, t]`, the last returned sample).
- `logp` — log-density at `x`.
- `trajectory` — D×T matrix produced by the most recent DEER solve.
- `tape` — noise tape used for that solve.
- `t` — index within `trajectory` of the last returned sample (1-indexed).
"""
struct DEERState{V<:AbstractVector, M<:AbstractMatrix}
    x::V
    logp::Float64
    trajectory::M
    tape::Vector{MALATapeElement{V}}
    t::Int
end

"""
One DEER sample: parameter vector `x` and its log-density `logp`.
"""
struct DEERTransition{V<:AbstractVector}
    x::V
    logp::Float64
end

function _build_mala_deer_rec(
    model::DensityModel, ε::Float64, tape::Vector{<:MALATapeElement}
)
    logp     = model.logdensity
    gradlogp = model.grad_logdensity

    step_fwd = (x, te) -> MALA.mala_step_taped(logp, gradlogp, x, ε, te.ξ, te.u)
    step_lin = (x, te, a) -> MALA.mala_step_surrogate(logp, gradlogp, x, ε, te.ξ, a)
    consts   = (x, te) -> (MALA.mala_accept_indicator(logp, gradlogp, x, ε, te.ξ, te.u),)

    return DEER.TapedRecursion(
        step_fwd, step_lin, tape;
        consts=consts, const_example=(0.0,),
    )
end

function _deer_solve_new_tape(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler,
    x0::AbstractVector,
)
    D    = model.dim
    T    = sampler.T
    tape = [MALATapeElement(randn(rng, D), rand(rng)) for _ in 1:T]
    rec  = _build_mala_deer_rec(model, sampler.epsilon, tape)
    S    = DEER.solve(
        rec, x0;
        tol_abs  = sampler.tol_abs,
        tol_rel  = sampler.tol_rel,
        maxiter  = sampler.maxiter,
        jacobian = sampler.jacobian,
        damping  = sampler.damping,
        probes   = sampler.probes,
        rng      = rng,
    )
    return S, tape
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler;
    initial_params=nothing,
    kwargs...,
)
    x0 = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, model.dim)
    end

    S, tape = _deer_solve_new_tape(rng, model, sampler, x0)

    x1    = S[:, 1]
    logp1 = Float64(model.logdensity(x1))
    trans = DEERTransition(x1, logp1)
    state = DEERState(x1, logp1, S, tape, 1)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::DEERSampler,
    state::DEERState;
    kwargs...,
)
    T      = sampler.T
    t_next = state.t + 1

    if t_next <= T
        # Consume the next cached sample from the trajectory.
        x_new    = state.trajectory[:, t_next]
        logp_new = Float64(model.logdensity(x_new))
        trans    = DEERTransition(x_new, logp_new)
        new_state = DEERState(x_new, logp_new, state.trajectory, state.tape, t_next)
        return trans, new_state
    else
        # Trajectory exhausted — re-solve with a fresh tape.
        x0          = state.trajectory[:, T]
        S_new, tape = _deer_solve_new_tape(rng, model, sampler, x0)
        x_new    = S_new[:, 1]
        logp_new = Float64(model.logdensity(x_new))
        trans    = DEERTransition(x_new, logp_new)
        new_state = DEERState(x_new, logp_new, S_new, tape, 1)
        return trans, new_state
    end
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:DEERTransition},
    model::DensityModel,
    sampler::DEERSampler,
    state::DEERState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    N = length(samples)
    D = model.dim

    names = if param_names !== nothing
        param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp]

    vals      = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 1)

    for i in 1:N
        vals[i, :] .= samples[i].x
        internals[i, 1] = samples[i].logp
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    BatchedDensityModel(logdensity_batch, grad_logdensity_batch, dim)

Wraps batched log-density and gradient functions for use with
`MALA.mala_step_batched`.

- `logdensity_batch(X::AbstractMatrix) -> AbstractVector` —
  `X` is D×N; returns a length-N vector of log-densities.
- `grad_logdensity_batch(X::AbstractMatrix) -> AbstractMatrix` —
  `X` is D×N; returns a D×N gradient matrix (column `n` = gradient for chain `n`).
- `dim::Int` — dimensionality D of the parameter space.

# GPU use
Pass `CuMatrix` inputs and implement the two functions to return `CuArray`s.
Requires `cholM=nothing` in `mala_step_batched` for fully on-device execution.

# Example
```julia
logp_b(X)      = vec(sum(x -> -0.5x^2, X; dims=1))   # standard normal, N chains
gradlogp_b(X)  = -X
bmodel = BatchedDensityModel(logp_b, gradlogp_b, 3)

X  = randn(3, 100)   # 100 chains of dimension 3
Xi = randn(3, 100)
u  = rand(100)
X_next, accepted = MALA.mala_step_batched(bmodel.logdensity_batch,
                                          bmodel.grad_logdensity_batch,
                                          X, 0.1, Xi, u)
```
"""
struct BatchedDensityModel{F,G} <: AbstractMCMC.AbstractModel
    logdensity_batch::F
    grad_logdensity_batch::G
    dim::Int
end

"""
    AdaptiveMALASampler(epsilon_init; n_warmup, target_accept, gamma, t0, kappa, cholM)

MALA sampler with automatic step-size adaptation via dual averaging
(Nesterov 2009, as used in NUTS — Hoffman & Gelman 2014).

During the first `n_warmup` steps the step size `ε` is adapted online to drive
the Metropolis acceptance rate toward `target_accept` (default 0.574, which is
the asymptotically optimal rate for MALA in high dimensions).  After warmup the
smoothed estimate `ε̄` is frozen and used for all remaining steps.

# Algorithm
At warmup step `m`, given current log-acceptance ratio `logα`:

    α   = min(1, exp(logα))
    H̄_m = (1 − 1/(m+t₀)) H̄_{m−1} + (1/(m+t₀)) (δ − α)
    log ε_m  = μ − √m/γ · H̄_m               (instantaneous)
    log ε̄_m  = m^(−κ) log ε_m + (1−m^(−κ)) log ε̄_{m−1}   (smoothed)

where μ = log(10 ε₀) is a fixed target.  After warmup `ε̄` is used.

# Keyword arguments
- `n_warmup` — adaptation steps (default 1000).
- `target_accept` — δ, desired acceptance rate (default 0.574).
- `gamma` — γ, regularisation strength (default 0.05).
- `t0` — stability offset (default 10.0).
- `kappa` — shrinkage exponent κ ∈ (0.5, 1] (default 0.75).
- `cholM` — optional Cholesky factor of a mass matrix `M` (default `nothing` = identity).

# MCMCChains output
The `Chains` object includes internals `[:logp, :accepted, :step_size, :is_warmup]`.
After warmup, `step_size` is constant (the frozen `ε̄`).

# Parallel chains
Works with `MCMCThreads()`.  R-hat and ESS are computed automatically by
`MCMCChains` from multi-chain output.

# Turing.jl / LogDensityProblems
Load `LogDensityProblems` (and optionally `LogDensityProblemsAD`) then use the
`DensityModel(ld)` constructor to wrap any `LogDensityProblems`-compatible model
(including Turing/DynamicPPL models) directly.
"""
struct AdaptiveMALASampler{CM} <: AbstractMCMC.AbstractSampler
    epsilon_init::Float64
    n_warmup::Int
    target_accept::Float64
    gamma::Float64
    t0::Float64
    kappa::Float64
    cholM::CM
end

function AdaptiveMALASampler(
    epsilon_init::Real;
    n_warmup::Int=1000,
    target_accept::Real=0.574,
    gamma::Real=0.05,
    t0::Real=10.0,
    kappa::Real=0.75,
    cholM=nothing,
)
    return AdaptiveMALASampler(
        Float64(epsilon_init), n_warmup,
        Float64(target_accept), Float64(gamma), Float64(t0), Float64(kappa),
        cholM,
    )
end

struct AdaptiveMALAState{V<:AbstractVector}
    x::V
    logp::Float64
    epsilon::Float64       # instantaneous step size ε_m
    epsilon_bar::Float64   # smoothed step size ε̄_m  (frozen after warmup)
    H_bar::Float64         # dual-average statistic H̄_m
    step::Int              # warmup step counter (0 = initialisation)
end

struct AdaptiveMALATransition{V<:AbstractVector}
    x::V
    logp::Float64
    accepted::Bool
    step_size::Float64   # ε used for this step
    is_warmup::Bool
end

function _dual_average_update(
    epsilon_init::Float64,
    epsilon_bar::Float64,
    H_bar::Float64,
    m::Int,
    logα::Float64,
    sampler::AdaptiveMALASampler,
)
    α   = min(1.0, exp(logα))
    δ   = sampler.target_accept
    γ   = sampler.gamma
    t0  = sampler.t0
    κ   = sampler.kappa
    μ   = log(10.0 * epsilon_init)   # fixed shrinkage target

    H_bar_new     = (1.0 - 1.0 / (m + t0)) * H_bar + (1.0 / (m + t0)) * (δ - α)
    log_ε         = μ - sqrt(Float64(m)) / γ * H_bar_new
    log_ε_bar_new = Float64(m)^(-κ) * log_ε + (1.0 - Float64(m)^(-κ)) * log(epsilon_bar)

    return exp(log_ε), exp(log_ε_bar_new), H_bar_new
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdaptiveMALASampler;
    initial_params=nothing,
    kwargs...,
)
    x = if initial_params !== nothing
        copy(initial_params)
    else
        randn(rng, model.dim)
    end
    logp_val = Float64(model.logdensity(x))
    trans    = AdaptiveMALATransition(x, logp_val, true, sampler.epsilon_init, true)
    state    = AdaptiveMALAState(x, logp_val, sampler.epsilon_init, sampler.epsilon_init, 0.0, 0)
    return trans, state
end

function AbstractMCMC.step(
    rng::Random.AbstractRNG,
    model::DensityModel,
    sampler::AdaptiveMALASampler,
    state::AdaptiveMALAState;
    kwargs...,
)
    D          = model.dim
    in_warmup  = state.step < sampler.n_warmup
    ε          = in_warmup ? state.epsilon : state.epsilon_bar

    ξ = randn(rng, D)
    u = rand(rng)

    x_next, accepted, logα = MALA.mala_step_with_logα(
        model.logdensity, model.grad_logdensity, state.x, ε, ξ, u;
        cholM=sampler.cholM,
    )

    logp_next = accepted ? Float64(model.logdensity(x_next)) : state.logp

    # Dual-average adaptation (only during warmup)
    m_new            = state.step + 1
    ε_new, ε_bar_new, H_bar_new = if in_warmup
        _dual_average_update(sampler.epsilon_init, state.epsilon_bar, state.H_bar, m_new, logα, sampler)
    else
        state.epsilon, state.epsilon_bar, state.H_bar
    end

    trans     = AdaptiveMALATransition(x_next, logp_next, accepted, ε, in_warmup)
    new_state = AdaptiveMALAState(x_next, logp_next, ε_new, ε_bar_new, H_bar_new, m_new)
    return trans, new_state
end

function AbstractMCMC.bundle_samples(
    samples::Vector{<:AdaptiveMALATransition},
    model::DensityModel,
    sampler::AdaptiveMALASampler,
    state::AdaptiveMALAState,
    ::Type{MCMCChains.Chains};
    param_names=nothing,
    kwargs...,
)
    N = length(samples)
    D = model.dim

    names = if param_names !== nothing
        param_names
    else
        [Symbol("x[$i]") for i in 1:D]
    end

    internal_names = [:logp, :accepted, :step_size, :is_warmup]

    vals      = Matrix{Float64}(undef, N, D)
    internals = Matrix{Float64}(undef, N, 4)

    for i in 1:N
        s = samples[i]
        vals[i, :]  .= s.x
        internals[i, 1] = s.logp
        internals[i, 2] = s.accepted   ? 1.0 : 0.0
        internals[i, 3] = s.step_size
        internals[i, 4] = s.is_warmup  ? 1.0 : 0.0
    end

    return MCMCChains.Chains(
        hcat(vals, internals),
        vcat(names, internal_names),
        Dict(:parameters => names, :internals => internal_names),
    )
end

"""
    DensityModel(ld)

Construct a `DensityModel` from any object `ld` that implements the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface,
i.e. provides `LogDensityProblems.logdensity`, `LogDensityProblems.logdensity_and_gradient`,
and `LogDensityProblems.dimension`.

Requires the `LogDensityProblems` package to be loaded:

```julia
using LogDensityProblems          # gradient-free: only logdensity used
using LogDensityProblemsAD        # or this, for AD-based gradients
```

# Turing.jl example
```julia
using Turing, LogDensityProblems, LogDensityProblemsAD, Mooncake

@model function mymodel(data)
    μ ~ Normal(0, 1)
    data ~ Normal(μ, 1)
end

ld  = DynamicPPL.LogDensityFunction(mymodel(obs))
ldg = LogDensityProblemsAD.ADgradient(Mooncake.Extras.MooncakeAD(), ld)
model = DensityModel(ldg)

chain = sample(model, AdaptiveMALASampler(0.1; n_warmup=500), 2000;
               chain_type=MCMCChains.Chains, progress=true)
```

This method is defined in the `LogDensityProblemsExt` extension and is only
available when `LogDensityProblems` has been loaded.
"""
function DensityModel end   # extended by LogDensityProblemsExt
