# [API Reference](@id reference)

This page documents all public types and functions exported by ParallelMCMC.jl.

## Model

```@docs
DensityModel
```

## Samplers

```@docs
MALASampler
AdaptiveMALASampler
ParallelMALASampler
```

## Internal types

These types appear in `MCMCChains` internals and in the `AbstractMCMC` state/transition protocol.  You generally do not need to construct them directly.

```@docs
MALATapeElement
MALAState
MALATransition
AdaptiveMALAState
AdaptiveMALATransition
ParallelMALAState
ParallelMALATransition
```

## Index

```@index
Pages = ["95-reference.md"]
```
