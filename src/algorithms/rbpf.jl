import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp

export RBPF

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
    extras::AbstractVector;
    callback = nothing,
    kwargs...
)
    state = initialise(rng, model, algo, extra0)
    ll = 0.0
    for (i, obs) in enumerate(observations)
        state, step_ll = step(rng, model, algo, i, state, obs, extras[i])
        ll += step_ll

        callback === nothing || callback(rng, model, algo, state, i; kwargs...)
    end
    return state, ll
end

function filter(
    model::HierarchicalSSM, algo::RBPF, observations::AbstractVector, extras::AbstractVector
)
    return filter(default_rng(), model, algo, observations, extras)
end
