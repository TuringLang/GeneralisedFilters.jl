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

function initialise(
    rng::AbstractRNG, model::HierarchicalSSM{T}, algo::RBPF; kwargs...
) where {T}
    N = algo.N
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Create containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    particles = Vector{RaoBlackwellisedContainer{outer_type,inner_type}}(undef, N)
    log_ws = zeros(T, N)

    # Initialise containers
    for i in 1:N
        x = simulate(rng, outer_dyn; kwargs...)
        z = initialise(inner_model, algo.inner_algo; new_outer=x, kwargs...)
        particles[i] = RaoBlackwellisedContainer(x, z)
    end

    return ParticleContainer(particles, log_ws)
end

function marginal_predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, state; kwargs...
)
    proposed_x = simulate(rng, model.outer_dyn, t, state.x; kwargs...)
    proposed_z = predict(
        rng,
        model.inner_model,
        algo.inner_algo,
        t,
        state.z;
        prev_outer=state.x,
        new_outer=proposed_x,
        kwargs...,
    )

    return RaoBlackwellisedContainer(proposed_x, proposed_z)
end

function marginal_update(
    model::HierarchicalSSM, algo::RBPF, t::Integer, state, obs; kwargs...
)
    filtered_z, log_increment = update(
        model.inner_model, algo.inner_algo, t, state.z, obs; new_outer=state.x, kwargs...
    )

    return RaoBlackwellisedContainer(state.x, filtered_z), log_increment
end

function predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, states; kwargs...
)
    states.proposed, states.ancestors = resample(rng, algo.resampler, states.filtered, algo)

    states.proposed.particles = map(
        x -> marginal_predict(rng, model, algo, t, x; kwargs...),
        states.filtered[states.ancestors],
    )

    return states
end

function update(
    model::HierarchicalSSM{T}, algo::RBPF, t::Integer, states, obs; kwargs...
) where {T}
    log_increments = similar(states.filtered.log_weights)
    for i in 1:(algo.N)
        states.filtered.particles[i].z, log_increments[i] = update(
            model.inner_model,
            algo.inner_algo,
            t,
            states.proposed.particles[i].z,
            obs;
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )

        states.filtered.particles[i].x = states.proposed.particles[i].x
    end

    ## TODO: make this also work...
    # result = map(
    #     x -> marginal_update(model, algo, t, x, obs; kwargs...),
    #     collect(states.proposed)
    # )

    states.filtered.log_weights = states.proposed.log_weights + log_increments

    return states, logmarginal(states, algo)
end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

struct BatchRBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractFilter
    inner_algo::F
    n_particles::Int
    resampler::RS
end
function BatchRBPF(
    inner_algo, n_particles; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    return BatchRBPF(inner_algo, n_particles, ESSResampler(threshold, resampler))
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

function initialise(
    rng::AbstractRNG, model::HierarchicalSSM{T}, algo::BatchRBPF; kwargs...
) where {T}
    N = algo.n_particles
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = batch_simulate(outer_dyn, N, kwargs...)
    zs = initialise(inner_model, algo.inner_algo; new_outer=xs, kwargs...)
    log_ws = CUDA.zeros(T, N)

    return RaoBlackwellisedParticleContainer(xs, zs, log_ws)
end

# TODO: use RNG
# TODO: include ref_state
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

    states.proposed, states.ancestors = resample(rng, filter.resampler, states.filtered)

    new_x = batch_simulate(outer_dyn, step, states.proposed.x_particles, N; kwargs...)
    states.proposed.z_particles = predict(
        inner_model,
        filter.inner_algo,
        step,
        states.proposed.z_particles;
        prev_outer=states.proposed.x_particles,
        new_outer=new_x,
        kwargs...,
    )
    states.proposed.x_particles = new_x

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
    states.filtered.z_particles, inner_lls = update(
        model.inner_model,
        filter.inner_algo,
        step,
        states.proposed.z_particles,
        obs;
        new_outer=states.proposed.x_particles,
        kwargs...,
    )
    states.filtered.x_particles = deepcopy(states.proposed.x_particles)
    states.filtered.log_weights = states.proposed.log_weights .+ inner_lls

    step_ll = (
        logsumexp(states.filtered.log_weights) - logsumexp(states.proposed.log_weights)
    )
    return states, step_ll
end
