export FFBS

abstract type AbstractSmoother <: AbstractSampler end

struct FFBS{T<:AbstractParticleFilter}
    filter::T
end

""" 
    smooth(rng::AbstractRNG, alg::AbstractSmooterh, model::AbstractStateSpaceModel, obs::AbstractVector, M::Integer; callback, kwargs...)
"""
function smooth end

struct WeightedParticleRecorderCallback{T,WT}
    particles::Array{T}
    log_weights::Array{WT}
end

function (callback::WeightedParticleRecorderCallback)(
    model, filter, step, states, data; kwargs...
)
    filtered_states = states.filtered
    callback.particles[step, :] = filtered_states.particles
    callback.log_weights[step, :] = filtered_states.log_weights
    return nothing
end

function gen_trajectory(
    rng::Random.AbstractRNG,
    model::StateSpaceModel,
    particles::AbstractMatrix{T},  # Need better container
    log_weights::AbstractMatrix{WT},
    forward_state,
    n_timestep::Int;
    kwargs...,
) where {T,WT}
    trajectory = Vector{T}(undef, n_timestep)
    trajectory[end] = forward_state
    for step in (n_timestep - 1):-1:1
        backward_weights = backward(
            model,
            step,
            trajectory[step + 1],
            particles[step, :],
            log_weights[step, :];
            kwargs...,
        )
        ancestor = rand(rng, Categorical(softmax(backward_weights)))
        trajectory[step] = particles[step, ancestor]
    end
    return trajectory
end

function backward(
    model::StateSpaceModel, step::Integer, state, particles::T, log_weights::WT; kwargs...
) where {T,WT}
    transitions = map(particles) do prev_state
        SSMProblems.logdensity(model.dyn, step, prev_state, state; kwargs...)
    end
    return log_weights + transitions
end

function sample(
    rng::Random.AbstractRNG,
    model::StateSpaceModel{T,LDT},
    alg::FFBS{<:BootstrapFilter{N}},
    obs::AbstractVector,
    M::Integer;
    callback=nothing,
    kwargs...,
) where {T,LDT,N}
    n_timestep = length(obs)
    recorder = WeightedParticleRecorderCallback(
        Array{eltype(model.dyn)}(undef, n_timestep, N), Array{T}(undef, n_timestep, N)
    )

    particles, _ = filter(rng, model, alg.filter, obs; callback=recorder, kwargs...)

    # Backward sampling - exact
    idx_ref = rand(rng, Categorical(weights(particles.filtered)), M)
    trajectories = Array{eltype(model.dyn)}(undef, n_timestep, M)

    trajectories[end, :] = particles.filtered[idx_ref]
    for j in 1:M
        trajectories[:, j] = gen_trajectory(
            rng,
            model,
            recorder.particles,
            recorder.log_weights,
            trajectories[end, j],
            n_timestep,
        )
    end
    return trajectories
end
