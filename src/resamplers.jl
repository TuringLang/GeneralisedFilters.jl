using Random
using Distributions

export Multinomial, Systematic, Metropolis, Rejection

abstract type AbstractResampler end

function resample(
    rng::AbstractRNG, resampler::AbstractResampler, states::ParticleState{PT,WT}
) where {PT,WT}
    weights = StatsBase.weights(states)
    idxs = sample_ancestors(rng, resampler, weights)

    new_state = ParticleState(
        states.particles[idxs], fill(-log(WT(length(states))), length(states))
    )

    return new_state, idxs
end

# TODO: combine this with above definition
function resample(
    rng::AbstractRNG,
    resampler::AbstractResampler,
    states::RaoBlackwellisedParticleState{T,M,ZT},
) where {T,M,ZT}
    weights = StatsBase.weights(states)
    idxs = sample_ancestors(rng, resampler, weights)

    new_state = RaoBlackwellisedParticleState(
        states.x_particles[:, idxs],
        states.z_particles[idxs],
        CUDA.fill(-log(T(length(states))), length(states)),
    )

    return new_state, idxs
end

## CONDITIONAL RESAMPLING ##################################################################

abstract type AbstractConditionalResampler <: AbstractResampler end

struct ESSResampler <: AbstractConditionalResampler
    threshold::Float64
    resampler::AbstractResampler
    function ESSResampler(threshold, resampler::AbstractResampler=Systematic())
        return new(threshold, resampler)
    end
end

function resample(
    rng::AbstractRNG, cond_resampler::ESSResampler, state::ParticleState{PT,WT}
) where {PT,WT}
    n = length(state)
    # TODO: computing weights twice. Should create a wrapper to avoid this
    weights = StatsBase.weights(state)
    ess = inv(sum(abs2, weights))
    @debug "ESS: $ess"

    if cond_resampler.threshold * n ≥ ess
        return resample(rng, cond_resampler.resampler, state)
    else
        return state, collect(1:n)
    end
end

# HACK: Likewise this should be removed. Even more so as it's identical, but needed to avoid
# method ambiguity
function resample(
    rng::AbstractRNG,
    cond_resampler::ESSResampler,
    state::RaoBlackwellisedParticleState{T,M,ZT},
) where {T,M,ZT}
    n = length(state)
    # TODO: computing weights twice. Should create a wrapper to avoid this
    weights = StatsBase.weights(state)
    ess = inv(sum(abs2, weights))
    @debug "ESS: $ess"
    println(ess)

    if cond_resampler.threshold * n ≥ ess
        return resample(rng, cond_resampler.resampler, state)
    else
        return state, collect(1:n)
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
    vs = n * cumsum(weights)
    u0 = rand(rng, WT)

    # initialize sampling algorithm
    a = Vector{Int64}(undef, n)
    idx = 1

    @inbounds for i in 1:n
        u = u0 + (i - 1)
        while vs[idx] <= u
            idx += 1
        end
        a[i] = idx
    end

    return a
end

# Following Code 8 of Murray et. al (2015)
function sample_ancestors(
    rng::AbstractRNG, ::Systematic, weights::CuVector{WT}, n::Int=length(weights)
) where {WT}
    u = rand(rng, WT)
    W = cumsum(weights)
    # TODO: assume weights sum to unity and document
    W_tot = CUDA.@allowscalar W[end]
    r = n * W
    return O = min.(n - 1, trunc.(Int32, r .+ u)) .+ 1
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
