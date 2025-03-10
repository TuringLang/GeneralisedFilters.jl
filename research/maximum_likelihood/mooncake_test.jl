using GeneralisedFilters
using SSMProblems
using LinearAlgebra
using Random

## TOY MODEL ###############################################################################

# this is taken from an example in Kalman.jl
function toy_model(θ::T) where {T<:Real}
    μ0 = T[1.0, 0.0]
    Σ0 = Diagonal(ones(T, 2))

    A = T[0.8 θ/2; -0.1 0.8]
    Q = Diagonal(T[0.2, 1.0])
    b = zeros(T, 2)

    H = Matrix{T}(I, 1, 2)
    R = Diagonal(T[0.2])
    c = zeros(T, 1)

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

# data generation process with small sample
rng = MersenneTwister(1234)
true_model = toy_model(1.0)
_, _, ys = sample(rng, true_model, 20)

## RUN MOONCKAE TESTS ######################################################################

using DifferentiationInterface
import Mooncake
using DistributionsAD

function build_objective(rng, θ, algo, data)
    _, ll = GeneralisedFilters.filter(rng, toy_model(θ[]), algo, data)
    return -ll
end

# kalman filter likelihood testing (works, but is slow)
logℓ1 = θ -> build_objective(rng, θ, KF(), ys)
Mooncake.TestUtils.test_rule(rng, logℓ1, [0.7]; is_primitive=false, debug_mode=true)

# bootstrap filter likelihood testing (shouldn't work)
logℓ2 = θ -> build_objective(rng, θ, BF(512), ys)
Mooncake.TestUtils.test_rule(rng, logℓ2, [0.7]; is_primitive=false, debug_mode=true)

## FOR USE WITH DIFFERENTIATION INTERFACE ##################################################

# data should be part of the objective, but be held constant by DifferentiationInterface
logℓ3 = (θ, data) -> build_objective(rng, θ, KF(), data)

# set the backend with default configuration
backend = AutoMooncake(; config=nothing)

# prepare the gradient for faster subsequent iteration
grad_prep = prepare_gradient(logℓ3, backend, [0.7], Constant(ys))

# evaluate gradients and iterate to show proof of concept
DifferentiationInterface.gradient(logℓ3, grad_prep, backend, [0.7], Constant(ys))
DifferentiationInterface.gradient(logℓ3, grad_prep, backend, [0.8], Constant(ys))
DifferentiationInterface.gradient(logℓ3, grad_prep, backend, [0.9], Constant(ys))
DifferentiationInterface.gradient(logℓ3, grad_prep, backend, [1.0], Constant(ys))
