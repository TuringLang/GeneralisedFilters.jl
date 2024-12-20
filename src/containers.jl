using DataStructures: Stack
using Random: rand

using AcceleratedKernels

## INTERMEDIATES ############################################################################

mutable struct ParticleIntermediate{DT,AT}
    proposed::DT
    filtered::DT
    ancestors::AT
end

mutable struct Intermediate{DT}
    proposed::DT
    filtered::DT
end

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
    return BatchGaussianDistribution(d.μs[:, [i]], d.Σs[:, :, [i]])
end

function Base.getindex(d::BatchGaussianDistribution, i::AbstractVector)
    return BatchGaussianDistribution(d.μs[:, i], d.Σs[:, :, i])
end
function Base.setindex!(d::BatchGaussianDistribution, value::BatchGaussianDistribution, i)
    d.μs[:, i] = value.μs
    d.Σs[:, :, i] = value.Σs
    return d
end
function expand(d::BatchGaussianDistribution, M::Integer)
    new_μs = CuArray(zeros(eltype(d.μs), size(d.μs, 1), M))
    new_Σs = CuArray(zeros(eltype(d.Σs), size(d.Σs, 1), size(d.Σs, 2), M))
    new_μs[:, 1:size(d.μs, 2)] = d.μs
    new_Σs[:, :, 1:size(d.Σs, 3)] = d.Σs
    return BatchGaussianDistribution(new_μs, new_Σs)
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

mutable struct RaoBlackwellisedParticle{XT,ZT}
    x_particles::XT
    z_particles::ZT
end

# TODO: this needs to be generalised to account for the flatten Levy SSM state
mutable struct RaoBlackwellisedParticleState{
    T,M<:CUDA.AbstractMemory,PT<:RaoBlackwellisedParticle
}
    particles::PT
    log_weights::CuArray{T,1,M}
end

StatsBase.weights(state::RaoBlackwellisedParticleState) = softmax(state.log_weights)
Base.length(state::RaoBlackwellisedParticleState) = length(state.log_weights)

# Allow particle to be get and set via tree_states[:, 1:N] = states
function Base.getindex(state::RaoBlackwellisedParticle, i)
    return RaoBlackwellisedParticle(state.x_particles[:, [i]], state.z_particles[i])
end
function Base.getindex(state::RaoBlackwellisedParticle, i::AbstractVector)
    return RaoBlackwellisedParticle(state.x_particles[:, i], state.z_particles[i])
end
function Base.setindex!(state::RaoBlackwellisedParticle, value::RaoBlackwellisedParticle, i)
    state.x_particles[:, i] = value.x_particles
    state.z_particles[i] = value.z_particles
    return state
end
Base.length(state::RaoBlackwellisedParticle) = size(state.x_particles, 2)

function expand(particles::CuArray{T,2,Mem}, M::Integer) where {T,Mem<:CUDA.AbstractMemory}
    new_particles = CuArray(zeros(eltype(particles), size(particles, 1), M))
    new_particles[:, 1:size(particles, 2)] = particles
    return new_particles
end

# Method for increasing size of particle container
function expand(p::RaoBlackwellisedParticle, M::Integer)
    new_x = expand(p.x_particles, M)
    new_z = expand(p.z_particles, M)
    return RaoBlackwellisedParticle(new_x, new_z)
end

"""
    RaoBlackwellisedParticleContainer
"""
mutable struct RaoBlackwellisedParticleContainer{T,M<:CUDA.AbstractMemory,PT}
    filtered::RaoBlackwellisedParticleState{T,M,PT}
    proposed::RaoBlackwellisedParticleState{T,M,PT}
    ancestors::CuArray{Int,1,M}

    function RaoBlackwellisedParticleContainer(
        x_particles::CuArray{T,2,M}, z_particles::ZT, log_weights::CuArray{T,1,M}
    ) where {T,M<:CUDA.AbstractMemory,ZT}
        init_particles = RaoBlackwellisedParticleState(
            RaoBlackwellisedParticle(x_particles, z_particles), log_weights
        )
        prop_particles = RaoBlackwellisedParticleState(
            RaoBlackwellisedParticle(similar(x_particles), deepcopy(z_particles)),
            CUDA.zeros(T, size(x_particles, 2)),
        )
        ancestors = CuArray(1:size(x_particles, 2))

        return new{T,M,typeof(RaoBlackwellisedParticle(x_particles, z_particles))}(
            init_particles, prop_particles, ancestors
        )
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
end

function ParticleContainer(
    initial_states::Vector{T}, log_weights::Vector{WT}
) where {T,WT<:Real}
    init_particles = ParticleState(initial_states, log_weights)
    prop_particles = ParticleState(similar(initial_states), zero(log_weights))
    return ParticleContainer{T,WT}(init_particles, prop_particles, eachindex(log_weights))
end

Base.collect(state::ParticleState) = state.particles
Base.length(state::ParticleState) = length(state.particles)
Base.keys(state::ParticleState) = LinearIndices(state.particles)

# not sure if this is kosher, since it doesn't follow the convention of Base.getindex
Base.@propagate_inbounds Base.getindex(state::ParticleState, i) = state.particles[i]
# Base.@propagate_inbounds Base.getindex(state::ParticleState, i::Vector{Int}) = state.particles[i]

function reset_weights!(state::ParticleState{T,WT}) where {T,WT<:Real}
    fill!(state.log_weights, zero(WT))
    return state.log_weights
end

function update_ref!(
    proposed::ParticleState, ref_state::Union{Nothing,AbstractVector}, step::Integer=0
)
    if !isnothing(ref_state)
        proposed.particles[1] = ref_state[step]
    end
    return proposed
end

function update_ref!(
    proposed::RaoBlackwellisedParticleState,
    ref_state::Union{Nothing,AbstractVector},
    step::Integer=0,
)
    if !isnothing(ref_state)
        CUDA.@allowscalar begin
            proposed.particles.x_particles[:, 1] = ref_state[step].x_particles
            proposed.particles.z_particles.μs[:, 1] = ref_state[step].z_particles.μs
            proposed.particles.z_particles.Σs[:, :, 1] = ref_state[step].z_particles.Σs
        end
    end
    return proposed
end

## DENSE PARTICLE STORAGE ##################################################################

struct DenseParticleContainer{T}
    particles::OffsetVector{Vector{T},Vector{Vector{T}}}
    ancestors::Vector{Vector{Int}}
end

function get_ancestry(container::DenseParticleContainer{T}, i::Integer) where {T}
    a = i
    v = Vector{T}(undef, length(container.particles))
    ancestry = OffsetVector(v, -1)
    for t in length(container.ancestors):-1:1
        ancestry[t] = container.particles[t][a]
        a = container.ancestors[t][a]
    end
    ancestry[0] = container.particles[0][a]
    return ancestry
end

"""
    DenseAncestorCallback

A callback for dense ancestry storage, which fills a `DenseParticleContainer`.
"""
struct DenseAncestorCallback{T}
    container::DenseParticleContainer{T}

    function DenseAncestorCallback(::Type{T}) where {T}
        particles = OffsetVector(Vector{T}[], -1)
        ancestors = Vector{Int}[]
        return new{T}(DenseParticleContainer(particles, ancestors))
    end
end

function (c::DenseAncestorCallback)(model, filter, states, data; kwargs...)
    push!(c.container.particles, deepcopy(states.filtered.particles))
    return nothing
end

function (c::DenseAncestorCallback)(model, filter, step, states, data; kwargs...)
    push!(c.container.particles, deepcopy(states.filtered.particles))
    push!(c.container.ancestors, deepcopy(states.ancestors))
    return nothing
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
    tree::ParticleTree{T}, states::Vector{T}, ancestors::AbstractVector{Int64}
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

function get_offspring(a::AbstractVector{Int64})
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

## GPU SPARSE PARTICLE STORAGE #############################################################

# TODO: make the state type more general
mutable struct ParallelParticleTree{ST,M<:CUDA.AbstractMemory}
    states::ST
    parents::CuVector{Int64,M}
    leaves::CuVector{Int64,M}
    offspring::CuVector{Int64,M}

    function ParallelParticleTree(states::ST, M::Integer) where {ST}
        if M < length(states)
            throw(ArgumentError("M must be greater than or equal to the number of states"))
        end

        parents = CUDA.zeros(Int64, M)
        offspring = CUDA.zeros(Int64, M)
        N = length(states)
        # tree_states = CuArray{T}(undef, size(states, 1), M)
        # tree_states[:, 1:N] = states
        states = expand(states, M)
        tree_states = states
        leaves = CuArray(1:N)
        return new{ST,CUDA.DeviceMemory}(tree_states, parents, leaves, offspring)
    end
end

function scatter!(r, p, q)
    return r[q] .= p
end

function gather!(r, p, q)
    return r .= p[q]
end

function update_offspring!(offspring, leaves, parents)
    AcceleratedKernels.foreachindex(leaves) do i
        j = leaves[i]
        while (j > 0) && (offspring[j] == 0)
            j = parents[j]
            if j > 0
                offspring[j] -= 1
            end
        end
    end
end

function insert!(tree::ParallelParticleTree, states, ancestors::CuVector{Int64})
    b = CuVector{Int64}(undef, length(ancestors))
    gather!(b, tree.leaves, ancestors)

    # Update offspring counts
    offspring = ancestors_to_offspring(ancestors)
    scatter!(tree.offspring, offspring, tree.leaves)

    # Prune tree
    update_offspring!(tree.offspring, tree.leaves, tree.parents)

    # Expand tree if necessary
    # TODO: can we combine this with z computation and update z if expanding?
    if sum(tree.offspring .== 0) < length(ancestors)
        @debug "expanding tree"
        expand!(tree)
    end
    z = cumsum(tree.offspring .== 0)

    # Insert new states
    new_leaves = searchsortedfirst(z, CuArray(1:length(tree.leaves)))
    scatter!(tree.parents, b, new_leaves)
    tree.states[new_leaves] = states
    tree.leaves .= new_leaves
    return tree
end

function expand!(tree::ParallelParticleTree{T}) where {T}
    M = length(tree.states)

    new_parents = CUDA.zeros(Int64, 2M)
    new_parents[1:length(tree.parents)] = tree.parents
    tree.parents = new_parents

    new_offspring = CUDA.zeros(Int64, 2M)
    new_offspring[1:length(tree.offspring)] = tree.offspring
    tree.offspring = new_offspring

    # new_states = CuArray(undef, size(tree.states, 1), 2 * M)
    # new_states = CuArray{T}(undef, size(tree.states, 1), 2 * M)
    # new_states[:, 1:M] = tree.states
    # tree.states = new_states
    tree.states = expand(tree.states, 2M)

    return tree
end

# Get ancestry of all particles
function get_ancestry(tree::ParallelParticleTree{ST}, T::Integer) where {ST}
    paths = OffsetVector(Vector{ST}(undef, T + 1), -1)
    parents = copy(tree.leaves)
    for t in T:-1:1
        paths[t] = tree.states[parents]
        gather!(parents, tree.parents, parents)
    end
    return paths
end

# Get ancestory of a single particle
function get_ancestry(
    container::ParallelParticleTree{ST}, i::Integer, T::Integer
) where {ST}
    path = OffsetVector(Vector{ST}(undef, T + 1), -1)
    CUDA.@allowscalar begin
        ancestor_index = container.leaves[i]
        for t in T:-1:0
            path[t] = container.states[ancestor_index]
            ancestor_index = container.parents[ancestor_index]
        end
        return path
    end
end

"""
    ParallelAncestorCallback

A callback for parallel sparse ancestry storage, which preallocates and returns a populated 
`ParallelParticleTree` object.
"""
# TODO: this should be initialised during inference so types don't need to be predetermined
struct ParallelAncestorCallback{T}
    tree::ParallelParticleTree{T}

    # function ParallelAncestorCallback(
    #     ::Type{T}, N::Integer, D::Integer, C::Real=1.0
    # ) where {T}
    #     M = floor(Int64, C * N * log(N))
    #     nodes = CuArray{T}(undef, D, N)
    #     return new{T}(ParallelParticleTree(nodes, M))
    # end
end

function (c::ParallelAncestorCallback)(model, filter, states, data; kwargs...)
    # Initialisation
    @inbounds c.tree.states[1:(filter.N)] = deepcopy(states.filtered.particles)
    return nothing
end

function (c::ParallelAncestorCallback)(model, filter, step, states, data; kwargs...)
    # TODO: this is a combined prune/insert step—split them up
    insert!(c.tree, states.proposed.particles, states.ancestors)
    return nothing
end

## ANCESTOR STORAGE CALLBACK ###############################################################

"""
    AncestorCallback

A callback for sparse ancestry storage, which preallocates and returns a populated 
`ParticleTree` object.
"""
struct AncestorCallback{T}
    tree::ParticleTree{T}
end

function AncestorCallback(::Type{T}, N::Integer, C::Real=1.0) where {T}
    M = floor(Int64, C * N * log(N))
    nodes = Vector{T}(undef, N)
    return new{T}(ParticleTree(nodes, M))
end

function (c::AncestorCallback)(model, filter, intermediate, data; kwargs...)
    # Initialisation
    @inbounds c.tree.states[1:(filter.N)] = deepcopy(intermediate.filtered.particles)
    return nothing
end

function (c::AncestorCallback)(model, filter, step, intermediate, data; kwargs...)
    prune!(c.tree, get_offspring(intermediate.ancestors))
    insert!(c.tree, intermediate.proposed.particles, intermediate.ancestors)
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
