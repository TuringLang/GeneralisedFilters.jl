using KalmanFilters: KalmanFilters
import LinearAlgebra: Cholesky, cholesky, diag, dot

export KalmanFilter, filter, SRKF

struct KalmanFilter <: FilteringAlgorithm end

function initialise(
    model::LinearGaussianStateSpaceModel{T}, filter::KalmanFilter, extra
) where {T}
    μ0, Σ0 = calc_initial(model.dyn, extra)
    return (μ=μ0, Σ=Σ0)
end

function predict(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::V, Σ::M},
    extra,
) where {T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    μ, Σ = state.μ, state.Σ
    A, b, Q = calc_params(model.dyn, step, extra)
    μ̂ = A * μ + b
    Σ̂ = A * Σ * A' + Q
    return (μ=μ̂, Σ=Σ̂)
end

function update(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::V, Σ::M},
    obs::Vector{T},
    extra,
) where {T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    μ, Σ = state.μ, state.Σ
    H, c, R = calc_params(model.obs, step, extra)

    # Update state
    m = H * μ + c
    y = obs - m
    S = H * Σ * H' + R
    K = Σ * H' / S
    μ̂ = μ + K * y
    Σ̂ = Σ - K * H * Σ

    # Compute log-likelihood
    S = (S + S') / 2  # HACK: ensure S is symmetric (due to numerical errors)
    ll = logpdf(MvNormal(m, S), obs)

    return (μ=μ̂, Σ=Σ̂), ll
end

function step(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::V, Σ::M},
    obs::Vector{T},
    extra,
) where {T,V<:AbstractVector{T},M<:AbstractMatrix{T}}
    state = predict(model, filter, step, state, extra)
    state, ll = update(model, filter, step, state, obs, extra)
    return state, ll
end

function filter(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    data::Vector{Vector{T}},
    extra0,
    extras,
) where {T}
    state = initialise(model, filter, extra0)
    states = Vector{@NamedTuple{μ::Vector{T}, Σ::Matrix{T}}}(undef, length(data))
    ll = 0.0
    for (i, obs) in enumerate(data)
        state, step_ll = step(model, filter, i, state, obs, extras[i])
        states[i] = state
        ll += step_ll
    end
    return states, ll
end

"""
    SRKF()

    A square-root Kalman filter.

    Implemented by wrapping KalmanFilters.jl.
"""
struct SRKF <: FilteringAlgorithm end

struct FactorisedGaussian{T<:Real,V<:AbstractVector{T},M<:Cholesky{T}}
    μ::V
    Σ::M
end

function initialise(model::LinearGaussianStateSpaceModel{T}, ::SRKF, extra) where {T}
    μ0, Σ0 = calc_initial(model.dyn, extra)
    return FactorisedGaussian(μ0, cholesky(Σ0))
end

function predict(
    model::LinearGaussianStateSpaceModel{T},
    ::SRKF,
    step::Integer,
    state::FactorisedGaussian{T},
    extra,
) where {T}
    (; μ, Σ) = state
    A, b, Q = calc_params(model.dyn, step, extra)
    !all(b .== 0) && error("Non-zero b not supported for SRKF")
    tu = KalmanFilters.time_update(μ, Σ, A, cholesky(Q))
    return FactorisedGaussian(
        KalmanFilters.get_state(tu), KalmanFilters.get_sqrt_covariance(tu)
    )
end

function update(
    model::LinearGaussianStateSpaceModel{T},
    ::SRKF,
    step::Integer,
    state::FactorisedGaussian{T},
    obs::Vector{T},
    extra,
) where {T}
    (; μ, Σ) = state
    H, c, R = calc_params(model.obs, step, extra)
    !all(c .== 0) && error("Non-zero c not supported for SRKF")
    mu = KalmanFilters.measurement_update(μ, Σ, obs, H, cholesky(R))
    # Note: since Cholesky L came from QR decomposition, it may not have positive diagonals
    ll = compute_ll(mu.innovation, mu.innovation_covariance)

    return FactorisedGaussian(
        KalmanFilters.get_state(mu), KalmanFilters.get_sqrt_covariance(mu)
    ),
    ll
end

# Manual Gaussian likelihood computation valid for non-positive diagonals of L
function compute_ll(ỹ, Σ::Cholesky)
    v = Σ.L \ ỹ
    logdet = 2 * sum(log ∘ abs, diag(Σ.L))  # take abs to handle non-positive diagonals
    return -0.5 * (dot(v, v) + logdet + length(ỹ) * log(2π))
end

function step(
    model::LinearGaussianStateSpaceModel{T},
    filter::SRKF,
    step::Integer,
    state::FactorisedGaussian{T},
    obs::Vector{T},
    extra,
) where {T}
    state = predict(model, filter, step, state, extra)
    state, ll = update(model, filter, step, state, obs, extra)
    return state, ll
end

function filter(
    model::LinearGaussianStateSpaceModel{T},
    filter::SRKF,
    data::Vector{Vector{T}},
    extra0,
    extras,
) where {T}
    state = initialise(model, filter, extra0)
    states = Vector{FactorisedGaussian{T}}(undef, length(data))
    ll = 0.0
    for (i, obs) in enumerate(data)
        state, step_ll = step(model, filter, i, state, obs, extras[i])
        states[i] = state
        ll += step_ll
    end
    return states, ll
end
