"""
Raw GPU DEER benchmark for Bayesian logistic regression.

This bypasses AbstractMCMC.sample(...) and measures the core DEER block solve:
    x0 -> build tape -> build TapedRecursion -> DEER.solve(rec, x0)

Optionally also benchmarks batched logdensity evaluation on the solved trajectory.

Goal:
    isolate whether the GPU DEER kernel itself is fast, independent of
    ParallelMALASampler / AbstractMCMC wrapper overhead.

Run from the benchmarks/ParallelMCMCBenchmarks directory:
    julia --project scripts/bench_raw_deer_logreg.jl
"""

using Random
using BenchmarkTools
using Printf
using Statistics

using ParallelMCMC
using ParallelMCMCBenchmarks
using CUDA

const BayesLogReg = ParallelMCMCBenchmarks.BayesLogReg

rng = MersenneTwister(20251231)

# Start small, then sweep upward if needed.
D = 10
N_data = 200

X_f32, y_f32, _ = BayesLogReg.make_data(rng, N_data, D)
X_f32 = Float32.(X_f32)
y_f32 = Float32.(y_f32)

CUDA.functional() || error("CUDA is not functional in this environment")

X_gpu = CUDA.CuMatrix(X_f32)
y_gpu = CUDA.CuVector(y_f32)

logp_gpu, gradlogp_gpu = BayesLogReg.make_problem(X_gpu, y_gpu)
logp_gpu_batch, gradlogp_gpu_batch = BayesLogReg.make_problem_batched(X_gpu, y_gpu)

model_gpu = DensityModel(
    logp_gpu,
    gradlogp_gpu,
    D;
    logdensity_batch=logp_gpu_batch,
    grad_logdensity_batch=gradlogp_gpu_batch,
)

# Core DEER settings to test.
T_vals = [8, 16, 32, 64]

# Keep these fixed initially.
epsilon  = 0.1f0
maxiter  = 50
tol_abs  = 1.0f-4
tol_rel  = 1.0f-3
damping  = 0.5f0
probes   = 1

x0_gpu = CUDA.zeros(Float32, D)

"""
Build exactly the same DEER problem that ParallelMALASampler uses internally,
but expose it directly for raw benchmarking.
"""
function build_raw_deer_problem(
    rng::Random.AbstractRNG,
    model::DensityModel,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
    maxiter::Int,
    tol_abs::Float32,
    tol_rel::Float32,
    damping::Float32,
    probes::Int,
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    FP = typeof(epsilon)
    D = model.dim

    # Reproduce the current tape construction logic used in interface.jl
    tape = map(1:T) do _
        ξ = copyto!(similar(x0, D), randn(rng, FP, D))
        ParallelMCMC.MALATapeElement(ξ, FP(rand(rng)))
    end

    rec = ParallelMCMC._build_mala_deer_rec(
        model,
        epsilon,
        tape,
        x0;
        cholM=cholM,
        backend=backend,
    )

    return rec
end

"""
Time one raw DEER block solve, returning the trajectory S (D×T).
"""
function solve_raw_deer_block(
    rng::Random.AbstractRNG,
    model::DensityModel,
    x0::AbstractVector;
    epsilon::Float32,
    T::Int,
    maxiter::Int,
    tol_abs::Float32,
    tol_rel::Float32,
    damping::Float32,
    probes::Int,
    cholM=nothing,
    backend=DEER.DEFAULT_BACKEND,
)
    rec = build_raw_deer_problem(
        rng, model, x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
        cholM=cholM,
        backend=backend,
    )

    S = DEER.solve(
        rec,
        x0;
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        maxiter=maxiter,
        jacobian=:stoch_diag,
        damping=damping,
        probes=probes,
        rng=rng,
    )
    return S
end

"""
Benchmark just the raw DEER solve.

Reports:
- block solve time
- implied throughput = T / solve_time
"""
function bench_raw_deer_solve(model, x0; T, reps)
    println("  [GPU] raw DEER solve, T=$T")

    # warmup / compile
    S_warm = solve_raw_deer_block(
        MersenneTwister(42),
        model,
        x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
    )
    CUDA.synchronize()

    b = @benchmark begin
        S = solve_raw_deer_block(
            MersenneTwister(42),
            $model,
            $x0;
            epsilon=$epsilon,
            T=$T,
            maxiter=$maxiter,
            tol_abs=$tol_abs,
            tol_rel=$tol_rel,
            damping=$damping,
            probes=$probes,
        )
        CUDA.synchronize()
        S
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println("\n")

    t_ms = median(b).time / 1e6
    samples_per_sec = T / (median(b).time / 1e9)
    @printf "    median solve time: %.3f ms\n" t_ms
    @printf "    implied throughput: %.1f samples/sec\n\n" samples_per_sec

    return b
end

"""
Benchmark batched logdensity over a solved block S.

This isolates how expensive it is to score an entire DEER trajectory at once.
"""
function bench_block_logdensity(model, x0; T, reps)
    model.logdensity_batch === nothing && error("model.logdensity_batch is required")

    S = solve_raw_deer_block(
        MersenneTwister(42),
        model,
        x0;
        epsilon=epsilon,
        T=T,
        maxiter=maxiter,
        tol_abs=tol_abs,
        tol_rel=tol_rel,
        damping=damping,
        probes=probes,
    )
    CUDA.synchronize()

    println("  [GPU] batched logdensity on solved block, T=$T")

    # warmup
    lp = model.logdensity_batch(S)
    CUDA.synchronize()

    b = @benchmark begin
        lp = $model.logdensity_batch($S)
        CUDA.synchronize()
        lp
    end samples=reps evals=1

    show(stdout, MIME("text/plain"), b)
    println("\n")

    t_ms = median(b).time / 1e6
    @printf "    median block logdensity time: %.3f ms\n\n" t_ms

    return b
end

println("=" ^ 72)
println("Raw GPU DEER benchmark")
println("Model: Bayesian logistic regression   D=$D   N_data=$N_data   Float32")
println("Measures raw block solve throughput, bypassing AbstractMCMC")
println("=" ^ 72)
println()

solve_results = Dict{Int,BenchmarkTools.Trial}()
logp_results  = Dict{Int,BenchmarkTools.Trial}()

for T in T_vals
    reps = T <= 16 ? 8 : T <= 32 ? 6 : 4
    solve_results[T] = bench_raw_deer_solve(model_gpu, x0_gpu; T=T, reps=reps)
    logp_results[T]  = bench_block_logdensity(model_gpu, x0_gpu; T=T, reps=reps)
end

println("=" ^ 72)
println("Summary")
println("=" ^ 72)
@printf "%-8s  %16s  %18s  %18s\n" "T" "solve ms" "samples/sec" "block logp ms"

for T in T_vals
    t_solve_ms = median(solve_results[T]).time / 1e6
    t_solve_s  = median(solve_results[T]).time / 1e9
    rate = T / t_solve_s
    t_logp_ms = median(logp_results[T]).time / 1e6
    @printf "%-8d  %16.3f  %18.1f  %18.3f\n" T t_solve_ms rate t_logp_ms
end

println()