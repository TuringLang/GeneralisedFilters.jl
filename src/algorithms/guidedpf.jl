export GuidedFilter, GPF, AbstractProposal

## PROPOSALS ###############################################################################
"""
    AbstractProposal
"""
abstract type AbstractProposal end

function SSMProblems.distribution(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return throw(
        MethodError(
            SSMProblems.distribution, (model, prop, step, state, observation, kwargs...)
        ),
    )
end

function SSMProblems.simulate(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    state,
    observation;
    kwargs...,
)
    return rand(
        rng, SSMProblems.distribution(model, prop, step, state, observation; kwargs...)
    )
end

function SSMProblems.logdensity(
    model::AbstractStateSpaceModel,
    prop::AbstractProposal,
    step::Integer,
    prev_state,
    new_state,
    observation;
    kwargs...,
)
    return logpdf(
        SSMProblems.distribution(model, prop, step, prev_state, observation; kwargs...),
        new_state,
    )
end

## GUIDED FILTERING ########################################################################

struct GuidedFilter{N,RS<:AbstractResampler,P<:AbstractProposal} <:
       AbstractParticleFilter{N}
    resampler::RS
    proposal::P
end

function GuidedFilter(
    N::Integer, proposal::P; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
) where {P<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return GuidedFilter{N,typeof(conditional_resampler),P}(conditional_resampler, proposal)
end

"""Shorthand for `GuidedFilter`"""
const GPF = GuidedFilter

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::GuidedFilter{N};
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
    filter::GuidedFilter,
    step::Integer,
    states::ParticleContainer{T},
    observation;
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    states.proposed, states.ancestors = resample(
        rng, filter.resampler, states.filtered, filter
    )
    states.proposed.particles = map(
        x -> SSMProblems.simulate(
            rng, model, filter.proposal, step, x, observation; kwargs...
        ),
        collect(states.proposed),
    )

    return update_ref!(states, ref_state, filter, step)
end

function update(
    model::StateSpaceModel{T},
    filter::GuidedFilter{N},
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
) where {T,N}
    # this is a little messy and may require a deepcopy
    particle_collection = zip(
        collect(states.proposed), deepcopy(states.filtered.particles[states.ancestors])
    )

    log_increments = map(particle_collection) do (new_state, prev_state)
        log_f = SSMProblems.logdensity(model.dyn, step, prev_state, new_state; kwargs...)
        log_g = SSMProblems.logdensity(model.obs, step, new_state, observation; kwargs...)
        log_q = SSMProblems.logdensity(
            model, filter.proposal, step, prev_state, new_state, observation; kwargs...
        )

        # println(log_f)

        (log_f + log_g - log_q)
    end

    # println(logsumexp(log_increments))

    states.filtered.log_weights = states.proposed.log_weights + log_increments
    states.filtered.particles = states.proposed.particles

    return states, logmarginal(states, filter)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::GuidedFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    proposed_state = predict(rng, model, alg, iter, state, observation; kwargs...)
    filtered_state, ll = update(model, alg, iter, proposed_state, observation; kwargs...)

    return filtered_state, ll
end

function reset_weights!(state::ParticleState{T,WT}, idxs, ::GuidedFilter) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state
end

function logmarginal(states::ParticleContainer, ::GuidedFilter)
    return logsumexp(states.filtered.log_weights) - logsumexp(states.proposed.log_weights)
end
