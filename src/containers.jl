using DataStructures: Stack
import Random: rand

## GAUSSIAN STATES #########################################################################

# TODO: add Kalman gain, innovation covariance, and residuals
mutable struct GaussianContainer{XT,ΣT}
    proposed::Gaussian{XT,ΣT}
    filtered::Gaussian{XT,ΣT}
end

mutable struct BatchGaussianDistribution{T,M<:CUDA.AbstractMemory}
    μs::CuArray{T,2,M}
    Σs::CuArray{T,3,M}
end
function Base.getindex(d::BatchGaussianDistribution, i)
    return BatchGaussianDistribution(d.μs[:, i], d.Σs[:, :, i])
end

## RAO-BLACKWELLISED STATES ################################################################

"""
    RaoBlackwellisedContainer

A container for Rao-Blackwellised states, composed of a marginalised state `z` (e.g. a
Gaussian or Categorical distribution) and a singular state `x`.
"""
mutable struct RaoBlackwellisedContainer{XT,ZT}
    x::XT
    z::ZT
end

# TODO: this needs to be generalised to account for the flatten Levy SSM state
mutable struct RaoBlackwellisedParticleState{T,M<:CUDA.AbstractMemory,ZT}
    x_particles::CuArray{T,2,M}
    z_particles::ZT
    log_weights::CuArray{T,1,M}
end

StatsBase.weights(state::RaoBlackwellisedParticleState) = softmax(state.log_weights)
Base.length(state::RaoBlackwellisedParticleState) = size(state.x_particles, 2)

"""
    RaoBlackwellisedParticleContainer
"""
mutable struct RaoBlackwellisedParticleContainer{T,M<:CUDA.AbstractMemory,ZT}
    filtered::RaoBlackwellisedParticleState{T,M,ZT}
    proposed::RaoBlackwellisedParticleState{T,M,ZT}
    ancestors::CuArray{Int,1,M}

    function RaoBlackwellisedParticleContainer(
        x_particles::CuArray{T,2,M}, z_particles::ZT, log_weights::CuArray{T,1,M}
    ) where {T,M<:CUDA.AbstractMemory,ZT}
        init_particles = RaoBlackwellisedParticleState(
            x_particles, z_particles, log_weights
        )
        prop_particles = RaoBlackwellisedParticleState(
            similar(x_particles), deepcopy(z_particles), CUDA.zeros(T, size(x_particles, 2))
        )
        ancestors = CuArray(1:size(x_particles, 2))
        return new{T,M,ZT}(init_particles, prop_particles, ancestors)
    end
end

## PARTICLES ###############################################################################

"""
    ParticleState

A container for particle filters which composes the weighted sample into a distibution-like
object, with the states (or particles) distributed accoring to their log-weights.
"""
mutable struct ParticleState{PT,WT<:Real}
    particles::Vector{PT}
    log_weights::Vector{WT}
end

StatsBase.weights(state::ParticleState) = softmax(state.log_weights)

"""
    ParticleContainer

A container for information passed through each iteration of an abstract particle filter,
composed of both proposed and filtered states, as well as the ancestor indices.
"""
mutable struct ParticleContainer{T,WT}
    filtered::ParticleState{T,WT}
    proposed::ParticleState{T,WT}
    ancestors::Vector{Int}

    function ParticleContainer(
        initial_states::Vector{T}, log_weights::Vector{WT}
    ) where {T,WT<:Real}
        init_particles = ParticleState(initial_states, log_weights)
        prop_particles = ParticleState(similar(initial_states), zero(log_weights))
        return new{T,WT}(init_particles, prop_particles, eachindex(log_weights))
    end
end

Base.collect(state::ParticleState) = state.particles
Base.length(state::ParticleState) = length(state.particles)
Base.keys(state::ParticleState) = LinearIndices(state.particles)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleState, i) = state.particles[i]
# Base.@propagate_inbounds Base.getindex(state::ParticleState, i::Vector{Int}) = state.particles[i]

function reset_weights!(state::ParticleState{T,WT}, idx, ::AbstractFilter) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state.log_weights
end

function logmarginal(states::ParticleContainer, ::AbstractFilter)
    return logsumexp(states.filtered.log_weights) - logsumexp(states.proposed.log_weights)
end

function update_ref!(
    pc::ParticleContainer{T},
    ref_state::Union{Nothing,AbstractVector{T}},
    ::AbstractFilter,
    step::Integer=0,
) where {T}
    # this comes from Nicolas Chopin's package particles
    if !isnothing(ref_state)
        pc.proposed[1] = ref_state[step + 1]
        pc.filtered[1] = ref_state[step + 1]
        pc.ancestors[1] = 1
    end
    return pc
end

## SPARSE PARTICLE STORAGE #################################################################

Base.append!(s::Stack, a::AbstractVector) = map(x -> push!(s, x), a)

"""
    ParticleTree

A sparse container for particle ancestry, which tracks the lineage of the filtered draws.

# Reference

Jacob, P., Murray L., & Rubenthaler S. (2015). Path storage in the particle 
filter [doi:10.1007/s11222-013-9445-x](https://dx.doi.org/10.1007/s11222-013-9445-x)
"""
mutable struct ParticleTree{T}
    states::Vector{T}
    parents::Vector{Int64}
    leaves::Vector{Int64}
    offspring::Vector{Int64}
    free_indices::Stack{Int64}

    function ParticleTree(states::Vector{T}, M::Integer) where {T}
        nodes = Vector{T}(undef, M)
        initial_free_indices = Stack{Int64}()
        append!(initial_free_indices, M:-1:(length(states) + 1))
        @inbounds nodes[1:length(states)] = states
        return new{T}(
            nodes, zeros(Int64, M), 1:length(states), zeros(Int64, M), initial_free_indices
        )
    end
end

Base.length(tree::ParticleTree) = length(tree.states)
Base.keys(tree::ParticleTree) = LinearIndices(tree.states)

function prune!(tree::ParticleTree, offspring::Vector{Int64})
    # insert new offspring counts
    setindex!(tree.offspring, offspring, tree.leaves)

    # update each branch
    @inbounds for i in eachindex(offspring)
        j = tree.leaves[i]
        while (j > 0) && (tree.offspring[j] == 0)
            push!(tree.free_indices, j)
            j = tree.parents[j]
            if j > 0
                tree.offspring[j] -= 1
            end
        end
    end
    return tree
end

function insert!(
    tree::ParticleTree{T}, states::Vector{T}, ancestors::AbstractVector{<:Integer}
) where {T}
    # parents of new generation
    parents = getindex(tree.leaves, ancestors)

    # ensure there are enough dead branches
    if (length(tree.free_indices) < length(ancestors))
        @debug "expanding tree"
        expand!(tree)
    end

    # find places for new states
    @inbounds for i in eachindex(states)
        tree.leaves[i] = pop!(tree.free_indices)
    end

    # insert new generation and update parent child relationships
    setindex!(tree.states, states, tree.leaves)
    setindex!(tree.parents, parents, tree.leaves)
    return tree
end

function expand!(tree::ParticleTree)
    M = length(tree)
    resize!(tree.states, 2 * M)

    # new allocations must be zero valued, this is not a perfect solution
    tree.parents = [tree.parents; zero(tree.parents)]
    tree.offspring = [tree.offspring; zero(tree.offspring)]
    append!(tree.free_indices, (2 * M):-1:(M + 1))
    return tree
end

function get_offspring(a::AbstractVector{<:Integer})
    offspring = zero(a)
    for i in a
        offspring[i] += 1
    end
    return offspring
end

function get_ancestry(tree::ParticleTree{T}) where {T}
    paths = Vector{Vector{T}}(undef, length(tree.leaves))
    @inbounds for (k, i) in enumerate(tree.leaves)
        j = tree.parents[i]
        xi = tree.states[i]

        xs = [xi]
        while j > 0
            push!(xs, tree.states[j])
            j = tree.parents[j]
        end
        paths[k] = reverse(xs)
    end
    return paths
end

function rand(rng::AbstractRNG, tree::ParticleTree, weights::AbstractVector{<:Real})
    b = randcat(rng, weights)
    leaf = tree.leaves[b]

    j = tree.parents[leaf]
    xi = tree.states[leaf]

    xs = [xi]
    while j > 0
        push!(xs, tree.states[j])
        j = tree.parents[j]
    end
    return reverse(xs)
end

## ANCESTOR STORAGE CALLBACK ###############################################################

"""
    AncestorCallback

A callback for sparse ancestry storage, which preallocates and returns a populated 
`ParticleTree` object.
"""
struct AncestorCallback{T}
    tree::ParticleTree{T}

    function AncestorCallback(::Type{T}, N::Integer, C::Real=1.0) where {T}
        M = floor(Int64, C * N * log(N))
        nodes = Vector{T}(undef, N)
        return new{T}(ParticleTree(nodes, M))
    end
end

function (c::AncestorCallback)(model, filter, step, states, data; kwargs...)
    if step == 1
        # this may be incorrect, but it is functional
        @inbounds c.tree.states[1:(filter.N)] = deepcopy(states.filtered.particles)
    end
    # TODO: when using non-stack version, may be more efficient to wait until storage full
    # to prune
    prune!(c.tree, get_offspring(states.ancestors))
    insert!(c.tree, states.filtered.particles, states.ancestors)
    return nothing
end

"""
    ResamplerCallback

A callback which follows the resampling indices over the filtering algorithm. This is more
of a debug tool and visualizer for various resapmling algorithms.
"""
struct ResamplerCallback
    tree::ParticleTree

    function ResamplerCallback(N::Integer, C::Real=1.0)
        M = floor(Int64, C * N * log(N))
        nodes = collect(1:N)
        return new(ParticleTree(nodes, M))
    end
end

function (c::ResamplerCallback)(model, filter, step, states, data; kwargs...)
    if step != 1
        prune!(c.tree, get_offspring(states.ancestors))
        insert!(c.tree, collect(1:(filter.N)), states.ancestors)
    end
    return nothing
end
