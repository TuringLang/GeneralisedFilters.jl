import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF, BatchRBPF

struct RBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractParticleFilter
    inner_algo::F
    N::Int
    resampler::RS
end

function RBPF(
    inner_algo::AbstractFilter,
    N::Integer;
    threshold::Real=1.0,
    resampler::AbstractResampler=Systematic(),
)
    return RBPF(inner_algo, N, ESSResampler(threshold, resampler))
end

function instantiate(::HierarchicalSSM{T}, filter::RBPF, initial; kwargs...) where {T}
    N = filter.N
    return ParticleIntermediate(initial, initial, Vector{Int}(undef, N))
end

function initialise(
    rng::AbstractRNG,
    model::HierarchicalSSM{T},
    algo::RBPF;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    particles = map(
        x -> RaoBlackwellisedParticle(
            simulate(rng, model.outer_dyn; kwargs...),
            initialise(model.inner_model, algo.inner_algo; new_outer=x, kwargs...),
        ),
        1:(algo.N),
    )
    log_ws = zeros(T, algo.N)

    return update_ref!(ParticleDistribution(particles, log_ws), ref_state)
end

function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    algo::RBPF,
    t::Integer,
    filtered;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    new_particles = map(
        x -> marginal_predict(rng, model, algo, t, x; kwargs...), filtered.particles
    )
    # Don't need to deep copy weights as filtered will be overwritten in the update step
    proposed = ParticleDistribution(new_particles, filtered.log_weights)

    return update_ref!(proposed, ref_state, t)
end

function marginal_predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, state; kwargs...
)
    proposed_x = simulate(rng, model.outer_dyn, t, state.x; kwargs...)
    proposed_z = predict(
        rng,
        model.inner_model,
        algo.inner_algo,
        t,
        state.z;
        prev_outer=state.x,
        new_outer=proposed_x,
        kwargs...,
    )

    return RaoBlackwellisedParticle(proposed_x, proposed_z)
end

function update(
    model::HierarchicalSSM{T}, algo::RBPF, t::Integer, proposed, obs; kwargs...
) where {T}
    log_increments = similar(proposed.log_weights)
    new_particles = deepcopy(proposed.particles)
    for i in 1:(algo.N)
        new_particles[i].z, log_increments[i] = update(
            model.inner_model,
            algo.inner_algo,
            t,
            proposed.particles[i].z,
            obs;
            new_outer=proposed.particles[i].x,
            kwargs...,
        )
    end

    ## TODO: make this also work...
    # result = map(
    #     x -> marginal_update(model, algo, t, x, obs; kwargs...),
    #     collect(states.proposed)
    # )

    new_weights = proposed.log_weights + log_increments
    filtered = ParticleDistribution(new_particles, new_weights)

    ll_increment = logsumexp(new_weights) - logsumexp(proposed.log_weights)

    return filtered, ll_increment
end

function marginal_update(
    model::HierarchicalSSM, algo::RBPF, t::Integer, state, obs; kwargs...
)
    filtered_z, log_increment = update(
        model.inner_model, algo.inner_algo, t, state.z, obs; new_outer=state.x, kwargs...
    )

    return RaoBlackwellisedParticle(state.x, filtered_z), log_increment
end

#################################
#### GPU-ACCELERATED VERSION ####
#################################

struct BatchRBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractParticleFilter
    inner_algo::F
    N::Int
    resampler::RS
end
function BatchRBPF(
    inner_algo, n_particles; threshold::Real=1.0, resampler::AbstractResampler=Systematic()
)
    return BatchRBPF(inner_algo, n_particles, ESSResampler(threshold, resampler))
end

function searchsorted!(ws_cdf, us, idxs)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:length(us)
        # Binary search
        left = 1
        right = length(ws_cdf)
        while left < right
            mid = (left + right) ÷ 2
            if ws_cdf[mid] < us[i]
                left = mid + 1
            else
                right = mid
            end
        end
        idxs[i] = left
    end
end

function instantiate(model::HierarchicalSSM, algo::BatchRBPF, initial; kwargs...)
    N = algo.N
    return ParticleIntermediate(initial, initial, CuArray{Int}(undef, N))
end

function initialise(
    rng::AbstractRNG,
    model::HierarchicalSSM{T},
    algo::BatchRBPF;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
) where {T}
    N = algo.N
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    xs = SSMProblems.batch_simulate(rng, outer_dyn, N; kwargs...)
    zs = initialise(inner_model, algo.inner_algo; new_outer=xs, kwargs...)
    log_ws = CUDA.zeros(T, N)

    return update_ref!(
        RaoBlackwellisedParticleDistribution(
            BatchRaoBlackwellisedParticles(xs, zs), log_ws
        ),
        ref_state,
    )
end

# TODO: use RNG
function predict(
    rng::AbstractRNG,
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    filtered::RaoBlackwellisedParticleDistribution;
    ref_state::Union{Nothing,AbstractVector}=nothing,
    kwargs...,
)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    new_xs = SSMProblems.batch_simulate(
        rng, outer_dyn, step, filtered.particles.xs; kwargs...
    )
    new_zs = predict(
        inner_model,
        filter.inner_algo,
        step,
        filtered.particles.zs;
        prev_outer=filtered.particles.xs,
        new_outer=new_xs,
        kwargs...,
    )
    # Don't need to deep copy weights as filtered will be overwritten in the update step
    proposed = RaoBlackwellisedParticleDistribution(
        BatchRaoBlackwellisedParticles(new_xs, new_zs), filtered.log_weights
    )

    # return states
    return update_ref!(proposed, ref_state, step)
end

function update(
    model::HierarchicalSSM,
    filter::BatchRBPF,
    step::Integer,
    proposed::RaoBlackwellisedParticleDistribution,
    obs;
    kwargs...,
)
    new_zs, inner_lls = update(
        model.inner_model,
        filter.inner_algo,
        step,
        proposed.particles.zs,
        obs;
        new_outer=proposed.particles.xs,
        kwargs...,
    )

    new_weights = proposed.log_weights + inner_lls
    # Don't need to deep copy particles as update will be overwritten in the next step
    filtered = RaoBlackwellisedParticleDistribution(
        BatchRaoBlackwellisedParticles(proposed.particles.xs, new_zs), new_weights
    )

    step_ll = logsumexp(filtered.log_weights) - logsumexp(proposed.log_weights)
    return filtered, step_ll
end
