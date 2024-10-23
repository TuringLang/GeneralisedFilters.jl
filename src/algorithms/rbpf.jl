import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF, BatchRBPF

struct RBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractFilter
    inner_algo::F
    N::Int
    resampler::RS
end

function RBPF(
    inner_algo::AbstractFilter,
    N::Integer;
    threshold::Real=1.0,
    resampler::AbstractResampler=Systematic(),
)
    return RBPF(inner_algo, N, ESSResampler(threshold, resampler))
end

function initialise(rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF; kwargs...)
    N = algo.N
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Create containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    particles = Vector{RaoBlackwellisedContainer{outer_type,inner_type}}(undef, N)
    log_ws = fill(-log(N), N)

    # Initialise containers
    for i in 1:N
        x = simulate(rng, outer_dyn; kwargs...)
        z = initialise(inner_model, algo.inner_algo; new_outer=x, kwargs...)
        particles[i] = RaoBlackwellisedContainer(x, z)
    end

    return ParticleContainer(particles, log_ws)
end

function predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, states; kwargs...
)
    states.proposed, states.ancestors = resample(rng, algo.resampler, states.filtered)

    for i in 1:(algo.N)
        prev_x = states.proposed.particles[i].x
        states.proposed[i].x = simulate(rng, model.outer_dyn, t, prev_x; kwargs...)

        prev_z = states.proposed.particles[i].z
        states.proposed.particles[i].z = predict(
            rng,
            model.inner_model,
            algo.inner_algo,
            t,
            prev_z;
            prev_outer=prev_x,
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )
    end

    return states
end

function update(
    model::HierarchicalSSM{T}, algo::RBPF, t::Integer, states, obs; kwargs...
) where {T}
    inner_lls = Vector{T}(undef, algo.N)
    for i in 1:(algo.N)
        states.filtered.particles[i].z, inner_ll = update(
            model.inner_model,
            algo.inner_algo,
            t,
            states.proposed.particles[i].z,
            obs;
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )
        inner_lls[i] = inner_ll

        states.filtered.particles[i].x = states.proposed.particles[i].x
    end

    states.filtered.log_weights = states.proposed.log_weights .+ inner_lls
    return states, logsumexp(inner_lls) - log(algo.N)
end

# function filter(
#     rng::AbstractRNG,
#     model::HierarchicalSSM,
#     algo::RBPF,
#     observations::AbstractVector,
#     extra0,
#     extras::AbstractVector,
# )
#     state = initialise(rng, model, algo, extra0)
#     ll = 0.0
#     for (i, obs) in enumerate(observations)
#         state, step_ll = step(rng, model, algo, i, state, obs, extras[i])
#         ll += step_ll
#     end
#     return state, ll
# end

# function filter(
#     model::HierarchicalSSM,
#     algo::RBPF,
#     observations::AbstractVector,
#     extra,
#     extras::AbstractVector,
# )
#     return filter(default_rng(), model, algo, observations, extra, extras)
# end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

struct BatchRBPF{F<:AbstractFilter} <: AbstractFilter
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

function initialise(rng::AbstractRNG, model::HierarchicalSSM, algo::BatchRBPF; kwargs...)
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = batch_simulate(outer_dyn, N, kwargs...)
    zs = initialise(inner_model, algo.inner_algo; new_outer=xs, kwargs...)
    log_ws = CUDA.fill(convert(Float32, -log(N)), N)

    return RaoBlackwellisedParticleContainer(xs, zs, log_ws)
end

resample(states::AbstractMatrix, idxs) = states[:, idxs]
# TODO: write a proper `resample` function (should probably be in Lévy SSM package)
# Note for now though that since Lévy SSM is independent, resampling isn't needed
resample(states, idxs) = states

# function step(model::HierarchicalSSM, algo::BatchRBPF, t::Integer, state, obs; kwargs...)
#     xs, zs, log_ws = state
#     N = algo.n_particles
#     outer_dyn, inner_model = model.outer_dyn, model.inner_model

#     # Optional resampling
#     weights = softmax(log_ws)
#     ess = 1 / sum(weights .^ 2)
#     if ess < algo.resample_threshold * N
#         cdf = cumsum(weights)
#         us = CUDA.rand(N)
#         idxs = CuArray{Int32}(undef, N)
#         @cuda threads = 256 blocks = 4096 searchsorted!(cdf, us, idxs)
#         # TODO: generalise this for non-`Vector` containers
#         xs = resample(xs, idxs)
#         # TODO: generalise this for other inner types
#         μs = zs.μs[:, idxs]
#         Σs = zs.Σs[:, :, idxs]
#         zs = (μs=μs, Σs=Σs)
#         log_ws .= convert(Float32, -log(N))
#     end

#     new_xs = batch_simulate(outer_dyn, t, xs, N, kwargs...)
#     new_extras = (prev_outer=xs, new_outer=new_xs)
#     inner_extra = isnothing(extra) ? new_extras : (; extra..., new_extras...)

#     zs, inner_lls = step(inner_model, algo.inner_algo, t, zs, obs, inner_extra)

#     log_ws += inner_lls

#     ll = logsumexp(inner_lls) - log(N)
#     return (new_xs, zs, log_ws), ll
# end

# Rewrite using `predict`/`update` and following RBPF format
# TODO: add RNG and ref_state
function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    states::RaoBlackwellisedParticleContainer;
    kwargs...,
)
    N = filter.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Skip resampling for now
    states.proposed = deepcopy(states.filtered)
    states.ancestors = CuArray(1:N)

    new_x = batch_simulate(outer_dyn, step, states.proposed.x_particles, N; kwargs...)
    states.filtered.z_particles = predict(
        inner_model,
        filter.inner_algo,
        step,
        states.proposed.z_particles;
        prev_outer=states.proposed.x_particles,
        new_outer=new_x,
        kwargs...,
    )
    states.filtered.x_particles = new_x

    return states
end

function update(
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    states::RaoBlackwellisedParticleContainer,
    obs;
    kwargs...,
)
    N = filter.n_particles
    states.filtered.z_particles, inner_lls = update(
        model.inner_model,
        filter.inner_algo,
        step,
        states.proposed.z_particles,
        obs;
        new_outer=states.proposed.x_particles,
        kwargs...,
    )
    states.filtered.x_particles = states.proposed.x_particles
    states.filtered.log_weights = states.proposed.log_weights .+ inner_lls

    return states, logsumexp(inner_lls) - log(N)
end
