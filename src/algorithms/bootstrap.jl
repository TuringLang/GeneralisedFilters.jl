export BootstrapFilter, BF

abstract type AbstractParticleFilter <: AbstractFilter end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractParticleFilter,
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
        rng, model, alg, iter, intermediate.proposed; ref_state=ref_state, kwargs...
    )
    # TODO: this is quite inelegant and should be refactored
    if !isnothing(ref_state)
        CUDA.@allowscalar intermediate.ancestors[1] = 1
    end

    intermediate.filtered, ll_increment = update(
        model, alg, iter, intermediate.proposed, observation; kwargs...
    )

    return intermediate, ll_increment
end

struct BootstrapFilter{RS<:AbstractResampler} <: AbstractParticleFilter
    N::Integer
    resampler::RS
end

"""Shorthand for `BootstrapFilter`"""
const BF = BootstrapFilter

function BootstrapFilter(
    N::Integer; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    conditional_resampler = ESSResampler(threshold, resampler)
    return BootstrapFilter{ESSResampler}(N, conditional_resampler)
end

function instantiate(
    ::StateSpaceModel{T}, filter::BootstrapFilter, initial; kwargs...
) where {T}
    N = filter.N
    return ParticleIntermediate(initial, deepcopy(initial), Vector{Int}(undef, N))
end

function initialise(
    rng::AbstractRNG,
    model::StateSpaceModel{T},
    filter::BootstrapFilter;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(x -> SSMProblems.simulate(rng, model.dyn; kwargs...), 1:(filter.N))
    weights = zeros(T, filter.N)

    return update_ref!(ParticleState(particles, weights), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::StateSpaceModel,
    filter::BootstrapFilter,
    step::Integer,
    filtered::ParticleState;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    new_particles = map(
        x -> SSMProblems.simulate(rng, model.dyn, step, x; kwargs...), collect(filtered)
    )
    proposed = ParticleState(new_particles, deepcopy(filtered.log_weights))

    return update_ref!(proposed, ref_state, step)
end

function update(
    model::StateSpaceModel{T},
    filter::BootstrapFilter,
    step::Integer,
    proposed::ParticleState,
    observation;
    kwargs...,
) where {T}
    log_increments = map(
        x -> SSMProblems.logdensity(model.obs, step, x, observation; kwargs...),
        collect(proposed),
    )

    new_weights = proposed.log_weights + log_increments
    filtered = ParticleState(deepcopy(proposed.particles), new_weights)

    ll_increment = logsumexp(filtered.log_weights) - logsumexp(proposed.log_weights)

    return filtered, ll_increment
end
