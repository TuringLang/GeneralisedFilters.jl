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

struct GuidedFilter{RS<:AbstractResampler,P<:AbstractProposal} <: AbstractParticleFilter
    N::Integer
    resampler::RS
    proposal::P
end

function GuidedFilter(
    N::Integer, proposal::PT; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
) where {PT<:AbstractProposal}
    conditional_resampler = ESSResampler(threshold, resampler)
    return GuidedFilter{ESSResampler,PT}(N, conditional_resampler, proposal)
end

function instantiate(
    ::StateSpaceModel{T}, filter::GuidedFilter, initial; kwargs...
) where {T}
    N = filter.N
    return ParticleIntermediate(initial, deepcopy(initial), Vector{Int}(undef, N))
end

"""Shorthand for `GuidedFilter`"""
const GPF = GuidedFilter

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::GuidedFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    weights = zeros(T, filter.N)

    return update_ref!(ParticleDistribution(particles, weights), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::GuidedFilter,
    step::Integer,
    filtered::ParticleDistribution,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    new_particles = map(
        x -> SSMProblems.simulate(
            rng, model, filter.proposal, step, x, observation; kwargs...
        ),
        collect(filtered)
    )
    proposed = ParticleDistribution(new_particles, deepcopy(filtered.log_weights))

    return update_ref!(proposed, ref_state, step)
end

function update(
    model::StateSpaceModel{T},
    filter::GuidedFilter,
    step::Integer,
    intermediate,
    observation;
    kwargs...,
) where {T}
    proposed = deepcopy(intermediate.proposed.particles)
    particle_collection = zip(
        proposed, deepcopy(intermediate.filtered.particles[intermediate.ancestors])
    )

    log_increments = map(particle_collection) do (new_state, prev_state)
        log_f = SSMProblems.logdensity(model.dyn, step, prev_state, new_state; kwargs...)
        log_g = SSMProblems.logdensity(model.obs, step, new_state, observation; kwargs...)
        log_q = SSMProblems.logdensity(
            model, filter.proposal, step, prev_state, new_state, observation; kwargs...
        )

        (log_f + log_g - log_q)
    end

    filtered = ParticleDistribution(
        proposed, intermediate.proposed.log_weights + log_increments
    )

    ll_increment  = logsumexp(intermediate.filtered.log_weights)
    ll_increment -= logsumexp(intermediate.proposed.log_weights)

    return filtered, ll_increment
end

# TODO: unify the particle interface between this and the bootstrap filter
function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::GuidedFilter,
    iter::Integer,
    intermediate,
    observation;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    intermediate.proposed, intermediate.ancestors = resample(
        rng, alg.resampler, intermediate.filtered
    )

    intermediate.proposed = predict(
        rng, model, alg, iter, intermediate.proposed, observation; ref_state, kwargs...
    )

    # TODO: this is quite inelegant and should be refactored
    if !isnothing(ref_state)
        CUDA.@allowscalar intermediate.ancestors[1] = 1
    end

    intermediate.filtered, ll_increment = update(
        model, alg, iter, intermediate, observation; kwargs...
    )

    return intermediate, ll_increment
end
