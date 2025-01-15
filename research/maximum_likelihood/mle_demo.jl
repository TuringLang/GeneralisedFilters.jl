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

# data generation process
rng = MersenneTwister(1234)
true_model = toy_model(1.0)
_, _, ys = sample(rng, true_model, 10000)

# evaluate and return the log evidence
function logℓ(θ, data)
    rng = MersenneTwister(1234)
    _, ll = GeneralisedFilters.filter(rng, toy_model(θ[]), KF(), data)
    return -ll
end

# check type stability (important for use with Enzyme)
@code_warntype logℓ([1.0], ys)

## MLE #####################################################################################

using DifferentiationInterface
using ForwardDiff
using Optimisers

# initial value
θ = [0.7]

# setup optimiser (feel free to use other backends)
state = Optimisers.setup(Optimisers.Descent(0.5), θ)
backend = AutoForwardDiff()
num_epochs = 1000

# prepare gradients for faster AD
grad_prep = prepare_gradient(logℓ, backend, θ, Constant(ys))
hess_prep = prepare_hessian(logℓ, backend, θ, Constant(ys))

for epoch in 1:num_epochs
    # calculate gradients
    val, ∇logℓ = DifferentiationInterface.value_and_gradient(
        logℓ, grad_prep, backend, θ, Constant(ys)
    )

    # adjust the learning rate for a hacky Newton's method
    H = DifferentiationInterface.hessian(logℓ, hess_prep, backend, θ, Constant(ys))
    Optimisers.update!(state, θ, inv(H)*∇logℓ)

    # stopping condition and printer
    (epoch % 5) == 1 && println("$(epoch-1):\t $(θ[])")
    if (∇logℓ'*∇logℓ) < 1e-12
        break
    end
end
