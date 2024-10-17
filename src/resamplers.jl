using Random
using Distributions

export Multinomial, Systematic, Metropolis, Rejection

abstract type AbstractResampler end

function resample(
    rng::AbstractRNG, resampler::AbstractResampler, states::ParticleState{PT,WT}
) where {PT,WT}
    weights = StatsBase.weights(states)
    idxs = sample_ancestors(rng, resampler, weights)

    return ParticleState(
        states.particles[idxs], fill(-log(WT(length(states))), length(states))
    )
end

## CONDITIONAL RESAMPLING ##################################################################

abstract type AbstractConditionalResampler end

struct ESSResampler <: AbstractConditionalResampler
    threshold::Float64
    resampler::AbstractResampler
    function ESSResampler(threshold, resampler::AbstractResampler=Systematic())
        return new(threshold, resampler)
    end
end

function resample(
    rng::AbstractRNG, cond_resampler::ESSResampler, state::ParticleState{T,WT}
) where {T,WT<:Real}
    n = length(state)
    weights = StatsBase.weights(state)
    ess = inv(sum(abs2, weights))
    @debug "ESS: $ess"

    if cond_resampler.threshold * n ≥ ess
        return resample(rng, cond_resampler.resampler, state)
    else
        return state
    end
end

## CATEGORICAL RESAMPLE ####################################################################

# this is adapted from AdvancedPS
function randcat(rng::AbstractRNG, weights::AbstractVector{WT}) where {WT<:Real}
    # pre-calculations
    @inbounds v = weights[1]
    u = rand(rng, WT)

    # initialize sampling algorithm
    n = length(weights)
    idx = 1

    while (v ≤ u) && (idx < n)
        idx += 1
        v += weights[idx]
    end

    return idx
end

## DOUBLE PRECISION STABLE ALGORITHMS ######################################################

struct Multinomial <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Multinomial, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    return rand(rng, Distributions.Categorical(weights), n)
end

struct Systematic <: AbstractResampler end

function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    @inbounds v = n * weights[1]
    u = rand(rng, WT)

    # initialize sampling algorithm
    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        while v < u
            idx += 1
            v += n * weights[idx]
        end
        a[i] = idx
        u += one(u)
    end

    return a
end

## SINGLE PRECISION STABLE ALGORITHMS ######################################################

struct Metropolis <: AbstractResampler
    ε::Float64
    function Metropolis(ε::Float64=0.01)
        return new(ε)
    end
end

# TODO: this should be done in the log domain and also parallelized
function sample_ancestors(
    rng::AbstractRNG,
    resampler::Metropolis,
    weights::AbstractVector{WT},
    n::Int64=length(weights);
) where {WT<:Real}
    # pre-calculations
    β = mean(weights)
    B = Int64(cld(log(resampler.ε), log(1 - β)))

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        k = i
        for _ in 1:B
            j = rand(rng, 1:n)
            v = weights[j] / weights[k]
            if rand(rng, WT) ≤ v
                k = j
            end
        end
        a[i] = k
    end

    return a
end

struct Rejection <: AbstractResampler end

# TODO: this should be done in the log domain and also parallelized
function sample_ancestors(
    rng::AbstractRNG, ::Rejection, weights::AbstractVector{WT}, n::Int64=length(weights)
) where {WT<:Real}
    # pre-calculations
    max_weight = maximum(weights)

    # initialize the algorithm
    a = Vector{Int64}(undef, n)

    @inbounds for i in 1:n
        j = i
        u = rand(rng)
        while u > weights[j] / max_weight
            j = rand(rng, 1:n)
            u = rand(rng, WT)
        end
        a[i] = j
    end

    return a
end
