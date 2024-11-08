export BootstrapFilter, BF

abstract type AbstractParticleFilter{N} <: AbstractFilter end

struct BootstrapFilter{N,RS<:AbstractResampler} <: AbstractParticleFilter{N}
    resampler::RS
end

function BootstrapFilter(
    N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    conditional_resampler = ESSResampler(threshold, resampler)
    return BootstrapFilter{N,typeof(conditional_resampler)}(conditional_resampler)
end

"""Shorthand for `BootstrapFilter`"""
const BF = BootstrapFilter

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::BootstrapFilter{N};
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {N,T}
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:N)
    initial_weights = zeros(T, N)

    return update_ref!(
        ParticleContainer(initial_states, initial_weights), ref_state, filter
    )
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    states::ParticleContainer{T};
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    states.proposed, states.ancestors = resample(
        rng, filter.resampler, states.filtered, filter
    )
    states.proposed.particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...),
        collect(states.proposed),
    )

    return update_ref!(states, ref_state, filter, step)
end

function update(
    model::StateSpaceModel{T},
    filter::BootstrapFilter{N},
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
) where {T,N}
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(states.proposed),
    )

    states.filtered.log_weights = states.proposed.log_weights + log_increments
    states.filtered.particles = states.proposed.particles

    return states, logmarginal(states, filter)
end

function reset_weights!(
    state::ParticleState{T,WT}, idxs, ::BootstrapFilter
) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state
end

function logmarginal(states::ParticleContainer, ::BootstrapFilter)
    return logsumexp(states.filtered.log_weights) - logsumexp(states.proposed.log_weights)
end