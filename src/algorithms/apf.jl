export AuxiliaryParticleFilter, APF

struct AuxiliaryParticleFilter{RS<:AbstractConditionalResampler} <: AbstractFilter
    N::Integer
    resampler::RS
    aux::Vector # Auxiliary weights
end

function AuxiliaryParticleFilter(
    N::Integer, threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    conditional_resampler = ESSResampler(threshold, resampler)
    return AuxiliaryParticleFilter(N, conditional_resampler, zeros(N))
end

const APF = AuxiliaryParticleFilter

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::AuxiliaryParticleFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    initial_states = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    initial_weights = fill(-log(T(filter.N)), filter.N)

    return update_ref!(ParticleContainer(initial_states, initial_weights), ref_state)
end

function update_weights!(
    rng::AbstractRNG, filter, model, step, states, observation; kwargs...
)
    simulation_weights = eta(rng, model, step, states, observation)
    return states.log_weights += simulation_weights
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::AuxiliaryParticleFilter,
    step::Integer,
    states::ParticleContainer{T},
    observation;
    ref_state::Union{Nothing,AbstractVector{T}}=nothing,
    kwargs...,
) where {T}
    # states = update_weights!(rng, filter.eta, model, step, states.filtered, observation; kwargs...)

    # Compute auxilary weights
    # POC: use the simplest approximation to the predictive likelihood
    # Ideally should be something like update_weights!(filter, ...)
    predicted = map(
        x -> mean(SSMProblems.distribution(model.dyn, step, x; kwargs...)),
        states.filtered.particles,
    )
    auxiliary_weights = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...), predicted
    )
    state.filtered.log_weights .+= auxiliary_weights
    filter.aux = auxiliary_weights

    states.proposed = resample(rng, filter.resampler, states.filtered, filter)
    states.proposed.particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...),
        states.proposed.particles,
    )

    return update_ref!(states, ref_state, step)
end

function update(
    model::StateSpaceModel{T},
    filter::AuxiliaryParticleFilter,
    step::Integer,
    states::ParticleContainer,
    observation;
    kwargs...,
) where {T}
    @debug "step $step"
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(states.proposed.particles),
    )

    states.filtered.log_weights = states.proposed.log_weights + log_increments
    states.filtered.particles = states.proposed.particles

    return (states, logsumexp(log_increments) - log(T(filter.N)))
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AuxiliaryParticleFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    proposed_state = predict(rng, model, alg, iter, state, observation; kwargs...)
    filtered_state, ll = update(model, alg, iter, proposed_state, observation; kwargs...)

    return filtered_state, ll
end

function reset_weights!(
    state::ParticleState{T,WT}, idxs, filter::AuxiliaryParticleFilter
) where {T,WT<:Real}
    # From Choping: An Introduction to sequential monte carlo, section 10.3.3
    state.log_weights = state.log_weights[idxs] - filter.aux[idxs]
    return state
end

function logmarginal(states::ParticleContainer, ::AuxiliaryParticleFilter)
    return logsumexp(states.filtered.log_weights) - logsumexp(states.proposed.log_weights)
end
