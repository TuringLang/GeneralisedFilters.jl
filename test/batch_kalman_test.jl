@testitem "Batch Kalman test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    using Random
    using SSMProblems

    using CUDA

    rng = StableRNG(1234)
    K = 3
    Dx = 4
    Dy = 3
    μ0s = [rand(rng, Dx) for _ in 1:K]
    Σ0s = [rand(rng, Dx, Dx) for _ in 1:K]
    Σ0s .= Σ0s .* transpose.(Σ0s)
    As = [rand(rng, Dx, Dx) for _ in 1:K]
    bs = [rand(rng, Dx) for _ in 1:K]
    Qs = [rand(rng, Dx, Dx) for _ in 1:K]
    Qs .= Qs .* transpose.(Qs)
    Hs = [rand(rng, Dy, Dx) for _ in 1:K]
    cs = [rand(rng, Dy) for _ in 1:K]
    Rs = [rand(rng, Dy, Dy) for _ in 1:K]
    Rs .= Rs .* transpose.(Rs)

    models = [
        create_homogeneous_linear_gaussian_model(
            μ0s[k], Σ0s[k], As[k], bs[k], Qs[k], Hs[k], cs[k], Rs[k]
        ) for k in 1:K
    ]

    T = 5
    Ys = [[rand(rng, Dy) for _ in 1:T] for _ in 1:K]

    outputs = [
        GeneralisedFilters.filter(rng, models[k], KalmanFilter(), Ys[k]) for k in 1:K
    ]

    states = first.(outputs)
    log_likelihoods = last.(outputs)

    struct BatchLinearGaussianDynamics{T,MT} <: LinearGaussianLatentDynamics{T}
        μ0s::CuArray{T,2,MT}
        Σ0s::CuArray{T,3,MT}
        As::CuArray{T,3,MT}
        bs::CuArray{T,2,MT}
        Qs::CuArray{T,3,MT}
    end

    function BatchLinearGaussianDynamics(
        μ0s::Vector{Vector{T}},
        Σ0s::Vector{Matrix{T}},
        As::Vector{Matrix{T}},
        bs::Vector{Vector{T}},
        Qs::Vector{Matrix{T}},
    ) where {T}
        μ0s = CuArray(stack(μ0s))
        Σ0s = CuArray(stack(Σ0s))
        As = CuArray(stack(As))
        bs = CuArray(stack(bs))
        Qs = CuArray(stack(Qs))
        return BatchLinearGaussianDynamics(μ0s, Σ0s, As, bs, Qs)
    end

    function GeneralisedFilters.batch_calc_μ0s(
        dyn::BatchLinearGaussianDynamics, ::Integer; kwargs...
    )
        return dyn.μ0s
    end
    function GeneralisedFilters.batch_calc_Σ0s(
        dyn::BatchLinearGaussianDynamics, ::Integer; kwargs...
    )
        return dyn.Σ0s
    end
    function GeneralisedFilters.batch_calc_As(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.As
    end
    function GeneralisedFilters.batch_calc_bs(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.bs
    end
    function GeneralisedFilters.batch_calc_Qs(
        dyn::BatchLinearGaussianDynamics, ::Integer, ::Integer; kwargs...
    )
        return dyn.Qs
    end

    struct BatchLinearGaussianObservations{T,MT} <: LinearGaussianObservationProcess{T}
        Hs::CuArray{T,3,MT}
        cs::CuArray{T,2,MT}
        Rs::CuArray{T,3,MT}
    end

    function BatchLinearGaussianObservations(
        Hs::Vector{Matrix{T}}, cs::Vector{Vector{T}}, Rs::Vector{Matrix{T}}
    ) where {T}
        Hs = CuArray(stack(Hs))
        cs = CuArray(stack(cs))
        Rs = CuArray(stack(Rs))
        return BatchLinearGaussianObservations(Hs, cs, Rs)
    end

    function GeneralisedFilters.batch_calc_Hs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.Hs
    end
    function GeneralisedFilters.batch_calc_cs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.cs
    end
    function GeneralisedFilters.batch_calc_Rs(
        obs::BatchLinearGaussianObservations, ::Integer, ::Integer; kwargs...
    )
        return obs.Rs
    end

    batch_model = GeneralisedFilters.StateSpaceModel(
        BatchLinearGaussianDynamics(μ0s, Σ0s, As, bs, Qs),
        BatchLinearGaussianObservations(Hs, cs, Rs),
    )

    # TODO: combine this into usual framework by allowing log_evidence to be a vector or
    # defininig a specific version for batch algorithms
    function GeneralisedFilters.filter(
        rng::AbstractRNG,
        model::SSMProblems.AbstractStateSpaceModel,
        alg::BatchKalmanFilter,
        observations::AbstractVector;
        callback=nothing,
        kwargs...,
    )
        initial = GeneralisedFilters.initialise(rng, model, alg; kwargs...)
        intermediate = GeneralisedFilters.instantiate(model, alg, initial; kwargs...)
        isnothing(callback) || callback(model, alg, intermediate, observations; kwargs...)
        log_evidence = CUDA.zeros(size(observations[1], 2))

        for t in eachindex(observations)
            intermediate, ll_increment = GeneralisedFilters.step(
                rng, model, alg, t, intermediate, observations[t]; callback, kwargs...
            )
            log_evidence .+= ll_increment
            isnothing(callback) ||
                callback(model, alg, t, intermediate, observations; kwargs...)
        end

        return intermediate.filtered, log_evidence
    end

    Ys_batch = Vector{Matrix{Float64}}(undef, T)
    for t in 1:T
        Ys_batch[t] = stack(Ys[k][t] for k in 1:K)
    end
    batch_output = GeneralisedFilters.filter(
        rng, batch_model, BatchKalmanFilter(K), Ys_batch
    )

    # println("Batch log-likelihood: ", batch_output[2])
    # println("Individual log-likelihoods: ", log_likelihoods)

    # println("Batch states: ", batch_output[1].μs')
    # println("Individual states: ", getproperty.(states, :μ))

    @test Array(batch_output[2])[end] .≈ log_likelihoods[end] rtol = 1e-5
    @test Array(batch_output[1].μs) ≈ stack(getproperty.(states, :μ)) rtol = 1e-5
end
