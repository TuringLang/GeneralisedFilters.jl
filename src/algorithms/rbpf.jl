import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF, BatchRBPF

struct RBPF{F<:FilteringAlgorithm} <: FilteringAlgorithm
    inner_algo::F
    n_particles::Int
    resample_threshold::Float64
end
RBPF(inner_algo::F, n_particles::Int) where {F} = RBPF(inner_algo, n_particles, 1.0)

function initialise(rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, extra)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Create containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    xs = Vector{outer_type}(undef, N)
    zs = Vector{inner_type}(undef, N)
    log_ws = fill(-log(N), N)

    # Initialise containers
    for i in 1:N
        xs[i] = simulate(rng, outer_dyn, extra)
        new_extra = (; new_outer=xs[i])
        inner_extra = isnothing(extra) ? new_extra : (; extra..., new_extra...)
        zs[i] = initialise(inner_model, algo.inner_algo, inner_extra)
    end

    return xs, zs, log_ws
end

function step(rng, model::HierarchicalSSM, algo::RBPF, t::Integer, state, obs, extra)
    xs, zs, log_ws = state

    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Optional resampling
    weights = Weights(softmax(log_ws))
    ess = 1 / sum(weights .^ 2)
    if ess < algo.resample_threshold * N
        idxs = sample(rng, 1:N, weights, N)
        xs .= xs[idxs]
        zs .= zs[idxs]
        log_ws .= fill(-log(N), N)
    end

    inner_lls = Vector{Float64}(undef, N)
    for i in 1:N
        prev_x = xs[i]
        xs[i] = simulate(rng, outer_dyn, t, prev_x, extra)

        new_extra = (prev_outer=prev_x, new_outer=xs[i])
        inner_extra = isnothing(extra) ? new_extra : (; extra..., new_extra...)

        zs[i], inner_ll = step(inner_model, algo.inner_algo, t, zs[i], obs, inner_extra)
        log_ws[i] = log_ws[i] + inner_ll
        inner_lls[i] = inner_ll
    end

    ll = logsumexp(inner_lls) - log(N)
    return (xs, zs, log_ws), ll
end

function filter(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    observations::AbstractVector,
    extra0,
    extras::AbstractVector,
)
    state = initialise(rng, model, algo, extra0)
    ll = 0.0
    for (i, obs) in enumerate(observations)
        state, step_ll = step(rng, model, algo, i, state, obs, extras[i])
        ll += step_ll
    end
    return state, ll
end

function filter(
    model::HierarchicalSSM,
    algo::RBPF,
    observations::AbstractVector,
    extra,
    extras::AbstractVector,
)
    return filter(default_rng(), model, algo, observations, extra, extras)
end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

struct BatchRBPF{F<:FilteringAlgorithm} <: FilteringAlgorithm
    inner_algo::F
    n_particles::Int
    resample_threshold::Float64
end
function BatchRBPF(inner_algo::F, n_particles::Int) where {F}
    return BatchRBPF(inner_algo, n_particles, 1.0)
end

function searchsorted!(ws_cdf, us, idxs)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(us)
        # Binary search
        left = 1
        right = length(ws_cdf)
        while left < right
            mid = (left + right) ÷ 2
            if ws_cdf[mid] < us[i]
                left = mid + 1
            else
                right = mid
            end
        end
        idxs[i] = left
    end
end

function initialise(model::HierarchicalSSM, algo::BatchRBPF, extra)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = batch_simulate(outer_dyn, extra, N)
    new_extra = (; new_outer=xs)
    inner_extra = isnothing(extra) ? new_extra : (; extra..., new_extra...)
    zs = initialise(inner_model, algo.inner_algo, inner_extra)
    log_ws = CUDA.fill(convert(Float32, -log(N)), N)

    return xs, zs, log_ws
end

resample(states::AbstractMatrix, idxs) = states[:, idxs]
# TODO: write a proper `resample` function (should probably be in Lévy SSM package)
# Note for now though that since Lévy SSM is independent, resampling isn't needed
resample(states, idxs) = states

function step(model::HierarchicalSSM, algo::BatchRBPF, t::Integer, state, obs, extra)
    xs, zs, log_ws = state
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Optional resampling
    weights = softmax(log_ws)
    ess = 1 / sum(weights .^ 2)
    if ess < algo.resample_threshold * N
        cdf = cumsum(weights)
        us = CUDA.rand(N)
        idxs = CuArray{Int32}(undef, N)
        @cuda threads = 256 blocks = 4096 searchsorted!(cdf, us, idxs)
        # TODO: generalise this for non-`Vector` containers
        xs = resample(xs, idxs)
        # TODO: generalise this for other inner types
        μs = zs.μs[:, idxs]
        Σs = zs.Σs[:, :, idxs]
        zs = (μs=μs, Σs=Σs)
        log_ws .= convert(Float32, -log(N))
    end

    new_xs = batch_simulate(outer_dyn, t, xs, extra, N)
    new_extras = (prev_outer=xs, new_outer=new_xs)
    inner_extra = isnothing(extra) ? new_extras : (; extra..., new_extras...)

    zs, inner_lls = step(inner_model, algo.inner_algo, t, zs, obs, inner_extra)

    log_ws += inner_lls

    ll = logsumexp(inner_lls) - log(N)
    return (new_xs, zs, log_ws), ll
end

function filter(
    model::HierarchicalSSM,
    algo::BatchRBPF,
    observations::AbstractVector,
    extra0,
    extras::AbstractVector,
)
    state = initialise(model, algo, extra0)
    ll = 0.0
    for (i, obs) in enumerate(observations)
        state, step_ll = step(model, algo, i, state, obs, extras[i])
        ll += step_ll
    end
    return state, ll
end
