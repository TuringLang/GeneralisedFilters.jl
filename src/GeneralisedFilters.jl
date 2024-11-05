module GeneralisedFilters

using AbstractMCMC: AbstractMCMC, AbstractSampler
import Distributions: MvNormal
import Random: AbstractRNG, default_rng, rand
using GaussianDistributions: pairs, Gaussian
using SSMProblems
using StatsBase

# TODO: heavy modules—move to extension
using CUDA
using NNlib

abstract type AbstractFilter <: AbstractSampler end

abstract type AbstractParticleFilter{N} <: AbstractFilter end

"""
    predict([rng,] model, alg, iter, state; kwargs...)

Propagate the filtered states forward in time.
"""
function predict end

"""
    update(model, alg, iter, state, observation; kwargs...)

Update beliefs on the propagated states.
"""
function update end

"""
    initialise([rng,] model, alg; kwargs...)

Propose an initial state distribution.
"""
function initialise end

"""
    step([rng,] model, alg, iter, state, observation; kwargs...)

Perform a combined predict and update call on a single iteration of the filter.
"""
function step end

"""
    reset_weights!(log_weights, filter)

Reset container log-weights after a resampling step
"""
function reset_weights! end

"""
    update_weights!
"""
function update_weights! end

function log_marginal end

function update_ref! end

function initialise(model, alg; kwargs...)
    return initialise(default_rng(), model, alg; kwargs...)
end

function predict(model, alg, step, states; kwargs...)
    return predict(default_rng(), model, alg, step, states; kwargs...)
end

function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    callback=nothing,
    kwargs...,
)
    states = initialise(rng, model, alg; kwargs...)
    log_evidence = zero(eltype(model))

    for t in eachindex(observations)
        states, log_marginal = step(
            rng, model, alg, t, states, observations[t]; callback, kwargs...
        )
        log_evidence += log_marginal
        isnothing(callback) || callback(model, alg, t, states, observations; kwargs...)
    end

    return states, log_evidence
end

function filter(
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    kwargs...,
)
    return filter(default_rng(), model, alg, observations; kwargs...)
end

function step(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    iter::Integer,
    state,
    observation;
    kwargs...,
)
    proposed_state = predict(rng, model, alg, iter, state; kwargs...)
    filtered_state, ll = update(model, alg, iter, proposed_state, observation; kwargs...)

    return filtered_state, ll
end

# Filtering utilities
include("containers.jl")
include("resamplers.jl")

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/bootstrap.jl")
include("algorithms/apf.jl")
include("algorithms/kalman.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

end
