module GeneralisedFilters

using AbstractMCMC: AbstractMCMC, AbstractSampler
import Distributions: MvNormal
import Random: AbstractRNG, default_rng, rand
using GaussianDistributions: pairs, Gaussian
using OffsetArrays
using SSMProblems
using StatsBase

# TODO: heavy modules—move to extension
using CUDA
using NNlib

## FILTERING BASE ##########################################################################

abstract type AbstractFilter <: AbstractSampler end

"""
    instantiate(model, alg, initial; kwargs...)

Create an intermediate storage object to store the proposed/filtered states at each step.
"""
function instantiate end

# Default method
function instantiate(model, alg, initial; kwargs...)
    return Intermediate(initial, deepcopy(initial))
end

"""
    initialise([rng,] model, alg; kwargs...)

Propose an initial state distribution.
"""
function initialise end

"""
    step([rng,] model, alg, iter, intermediate, observation; kwargs...)

Perform a combined predict and update call of the filtering on the intermediate storage.
"""
function step end

"""
    predict([rng,] model, alg, iter, filtered; kwargs...)

Propagate the filtered distribution forward in time.
"""
function predict end

"""
    update(model, alg, iter, proposed, observation; kwargs...)

Update beliefs on the propagated distribution given an observation.
"""
function update end

function initialise(model, alg; kwargs...)
    return initialise(default_rng(), model, alg; kwargs...)
end

function predict(model, alg, step, filtered; kwargs...)
    return predict(default_rng(), model, alg, step, filtered; kwargs...)
end

function filter(
    rng::AbstractRNG,
    model::AbstractStateSpaceModel,
    alg::AbstractFilter,
    observations::AbstractVector;
    callback=nothing,
    kwargs...,
)
    initial = initialise(rng, model, alg; kwargs...)
    intermediate = instantiate(model, alg, initial; kwargs...)
    isnothing(callback) || callback(model, alg, intermediate, observations; kwargs...)
    log_evidence = zero(eltype(model))

    for t in eachindex(observations)
        intermediate, ll_increment = step(
            rng, model, alg, t, intermediate, observations[t]; callback, kwargs...
        )
        log_evidence += ll_increment
        isnothing(callback) ||
            callback(model, alg, t, intermediate, observations; kwargs...)
    end

    return intermediate.filtered, log_evidence
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
    intermediate,
    observation;
    kwargs...,
)
    intermediate.proposed = predict(rng, model, alg, iter, intermediate.filtered; kwargs...)
    intermediate.filtered, ll_increment = update(
        model, alg, iter, intermediate.proposed, observation; kwargs...
    )

    return intermediate, ll_increment
end

## SMOOTHING BASE ##########################################################################

abstract type AbstractSmoother <: AbstractSampler end

# function smooth end
# function backward end

# Filtering utilities
include("containers.jl")
include("resamplers.jl")

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/bootstrap.jl")
include("algorithms/kalman.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

end
