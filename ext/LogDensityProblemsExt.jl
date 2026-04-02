module LogDensityProblemsExt

using ParallelMCMC
import LogDensityProblems

"""
    DensityModel(ld)

Construct a `DensityModel` from any object implementing the
[LogDensityProblems](https://github.com/tpapp/LogDensityProblems.jl) interface.

`ld` must support:
- `LogDensityProblems.capabilities(ld)` returning at least
  `LogDensityProblems.LogDensityOrder{1}` (i.e. gradient available).
- `LogDensityProblems.dimension(ld)` → `Int`
- `LogDensityProblems.logdensity_and_gradient(ld, x)` → `(logp, grad)`

# Turing.jl / DynamicPPL example
```julia
using Turing, LogDensityProblems, LogDensityProblemsAD, Mooncake, ParallelMCMC, MCMCChains

@model function mymodel(y)
    μ ~ Normal(0, 1)
    y ~ Normal(μ, 0.5)
end

obs = 1.5
ld  = DynamicPPL.LogDensityFunction(mymodel(obs))
ldg = LogDensityProblemsAD.ADgradient(Mooncake.Extras.MooncakeAD(), ld)

model = DensityModel(ldg)
chain = sample(model, AdaptiveMALASampler(0.3; n_warmup=500), 2_000;
               chain_type=MCMCChains.Chains, progress=true)
```
"""
function ParallelMCMC.DensityModel(ld)
    caps = LogDensityProblems.capabilities(ld)
    caps isa LogDensityProblems.LogDensityOrder{0} &&
        error("LogDensityProblems model must support gradients (LogDensityOrder{1} or higher). " *
              "Wrap it with LogDensityProblemsAD.ADgradient first.")

    dim = LogDensityProblems.dimension(ld)

    logp(x) = LogDensityProblems.logdensity(ld, x)

    function gradlogp(x)
        _, g = LogDensityProblems.logdensity_and_gradient(ld, x)
        return g
    end

    return ParallelMCMC.DensityModel(logp, gradlogp, dim)
end

end # module
