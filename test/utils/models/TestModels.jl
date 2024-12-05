"""Definitions and generators of state space models used within unit tests."""

@testmodule TestModels begin
    using CUDA
    using LinearAlgebra
    using NNlib
    using Random

    using GeneralisedFilters
    using SSMProblems

    include("linear_gaussian.jl")
    include("dummy_linear_gaussian.jl")

    function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
        Σ = rand(rng, T, d, d)
        return Σ * Σ'
    end
end
