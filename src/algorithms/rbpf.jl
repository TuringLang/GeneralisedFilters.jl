import LinearAlgebra: I
import Distributions: logpdf
import LogExpFunctions: softmax, logsumexp
import StatsBase: Weights

export RBPF

struct RBPF{F<:AbstractFilter,RS<:AbstractResampler} <: AbstractFilter
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

function initialise(rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF; kwargs...)
    N = algo.N
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    # Create containers
    outer_type, inner_type = eltype(outer_dyn), rb_eltype(inner_model)
    particles = Vector{RaoBlackwellisedContainer{outer_type,inner_type}}(undef, N)
    log_ws = fill(-log(N), N)

    # Initialise containers
    for i in 1:N
        x = simulate(rng, outer_dyn; kwargs...)
        z = initialise(inner_model, algo.inner_algo; new_outer=x, kwargs...)
        particles[i] = RaoBlackwellisedContainer(x, z)
    end

    return ParticleContainer(particles, log_ws)
end

function predict(
    rng::AbstractRNG, model::HierarchicalSSM, algo::RBPF, t::Integer, states; kwargs...
)
    states.proposed, states.ancestors = resample(rng, algo.resampler, states.filtered, algo)

    for i in 1:(algo.N)
        prev_x = states.proposed.particles[i].x
        states.proposed[i].x = simulate(rng, model.outer_dyn, t, prev_x; kwargs...)

        prev_z = states.proposed.particles[i].z
        states.proposed.particles[i].z = predict(
            rng,
            model.inner_model,
            algo.inner_algo,
            t,
            prev_z;
            prev_outer=prev_x,
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )
    end

    return states
end

function update(
    model::HierarchicalSSM{T}, algo::RBPF, t::Integer, states, obs; kwargs...
) where {T}
    inner_lls = Vector{T}(undef, algo.N)
    for i in 1:(algo.N)
        states.filtered.particles[i].z, inner_ll = update(
            model.inner_model,
            algo.inner_algo,
            t,
            states.proposed.particles[i].z,
            obs;
            new_outer=states.proposed.particles[i].x,
            kwargs...,
        )
        inner_lls[i] = inner_ll

        states.filtered.particles[i].x = states.proposed.particles[i].x
    end

    states.filtered.log_weights = states.proposed.log_weights .+ inner_lls

    return states, logsumexp(inner_lls) - log(algo.N)
end
