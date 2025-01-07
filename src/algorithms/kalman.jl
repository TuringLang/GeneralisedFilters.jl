export KalmanFilter, filter, BatchKalmanFilter
using GaussianDistributions

export KalmanFilter, KF, KalmanSmoother, KS

struct KalmanFilter <: AbstractFilter end

KF() = KalmanFilter()

function initialise(
    rng::AbstractRNG, model::LinearGaussianStateSpaceModel, filter::KalmanFilter; kwargs...
)
    μ0, Σ0 = calc_initial(model.dyn; kwargs...)
    return Gaussian(μ0, Matrix(Σ0))
end

function predict(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel,
    filter::KalmanFilter,
    step::Integer,
    filtered::Gaussian;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(filtered)
    A, b, Q = calc_params(model.dyn, step; kwargs...)
    return Gaussian(A * μ + b, A * Σ * A' + Q)
end

function update(
    model::LinearGaussianStateSpaceModel,
    filter::KalmanFilter,
    step::Integer,
    proposed::Gaussian,
    obs::AbstractVector;
    kwargs...,
)
    μ, Σ = GaussianDistributions.pair(proposed)
    H, c, R = calc_params(model.obs, step; kwargs...)

    # Update state
    m = H * μ + c
    y = obs - m
    S = H * Σ * H' + R
    K = Σ * H' / S

    # HACK: force the covariance to be positive definite
    S = (S + S') / 2

    filtered = Gaussian(μ + K * y, Σ - K * H * Σ)

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, S), obs)

    return filtered, ll
end

struct BatchKalmanFilter <: AbstractBatchFilter
    batch_size::Int
end

function initialise(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter;
    kwargs...,
) where {T}
    μ0s, Σ0s = batch_calc_initial(model.dyn, algo.batch_size; kwargs...)
    return BatchGaussianDistribution(μ0s, Σ0s)
end

function predict(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter,
    step::Integer,
    state::BatchGaussianDistribution;
    kwargs...,
) where {T}
    μs, Σs = state.μs, state.Σs
    As, bs, Qs = batch_calc_params(model.dyn, step, algo.batch_size; kwargs...)
    μ̂s = NNlib.batched_vec(As, μs) .+ bs
    Σ̂s = NNlib.batched_mul(NNlib.batched_mul(As, Σs), NNlib.batched_transpose(As)) .+ Qs
    return BatchGaussianDistribution(μ̂s, Σ̂s)
end

function invert_innovation(S)
    if size(S, 1) == 2
        return _invert_innovation_analytic(S)
    else
        return _invert_innovation(S)
    end
end

function _invert_innovation_analytic(S)
    # Analytic Cholesky for 2x2 matrix
    d_S = CUDA.zeros(Float32, size(S))
    d_S[1, 1, :] .= sqrt.(S[1, 1, :])
    d_S[2, 1, :] .= S[2, 1, :] ./ d_S[1, 1, :]
    d_S[1, 2, :] .= d_S[2, 1, :]
    d_S[2, 2, :] .= sqrt.(S[2, 2, :] .- d_S[2, 1, :] .^ 2)
    # Analytic inverse of 2x2 matrix
    S_inv = CUDA.zeros(Float32, size(S))
    S_det = S[1, 1, :] .* S[2, 2, :] .- S[2, 1, :] .* S[1, 2, :]
    S_inv[1, 1, :] .= S[2, 2, :]
    S_inv[2, 2, :] .= S[1, 1, :]
    S_inv[2, 1, :] .= -S[2, 1, :]
    S_inv[1, 2, :] .= -S[1, 2, :]
    S_inv ./= reshape(S_det, 1, 1, :)

    diags = CuArray{Float32}(undef, size(S, 1), size(S, 3))
    for i in 1:size(S, 1)
        diags[i, :] .= d_S[i, i, :]
    end
    log_dets = 2 * sum(log ∘ abs, diags; dims=1)

    return S_inv, log_dets
end

function _invert_innovation(S)
    # LU decomposition to compute S^{-1}
    # TODO: Replace with custom fast Cholesky kernel
    d_ipiv, _, d_S = CUDA.CUBLAS.getrf_strided_batched(S, true)
    S_inv = CUDA.similar(S)
    # TODO: This fails when D_obs > D_inner since S is not invertible
    CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)

    diags = CuArray{eltype(S)}(undef, size(S, 1), size(S, 3))
    for i in 1:size(S, 1)
        diags[i, :] .= d_S[i, i, :]
    end
    # L has ones on the diagonal so we can just multiply the diagonals of U
    # Since we're using pivoting, diagonal entries may be negative, so we take the absolute value
    log_dets = sum(log ∘ abs, diags; dims=1)

    return S_inv, log_dets
end

function update(
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter,
    step::Integer,
    state::BatchGaussianDistribution,
    obs;
    kwargs...,
) where {T}
    μs, Σs = state.μs, state.Σs
    Hs, cs, Rs = batch_calc_params(model.obs, step, algo.batch_size; kwargs...)

    m = NNlib.batched_vec(Hs, μs) .+ cs
    y_res = cu(obs) .- m
    S = NNlib.batched_mul(NNlib.batched_mul(Hs, Σs), NNlib.batched_transpose(Hs)) .+ Rs

    ΣH_T = NNlib.batched_mul(Σs, NNlib.batched_transpose(Hs))

    S_inv, log_dets = invert_innovation(S)

    K = NNlib.batched_mul(ΣH_T, S_inv)

    μ_filt = μs .+ NNlib.batched_vec(K, y_res)
    Σ_filt = Σs .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs))

    y = cu(obs)

    inv_term = NNlib.batched_vec(S_inv, y .- m)
    log_likes =
        -T(0.5) * NNlib.batched_vec(reshape(y .- m, 1, size(y, 1), size(S, 3)), inv_term)
    D = size(y, 1)
    log_likes = log_likes .- T(0.5) * (log_dets .+ D * log(T(2π)))

    # HACK: only errors seems to be from numerical stability so will just overwrite
    log_likes[isnan.(log_likes)] .= -Inf

    return BatchGaussianDistribution(μ_filt, Σ_filt), dropdims(log_likes; dims=1)
end

## KALMAN SMOOTHER #########################################################################

struct KalmanSmoother <: AbstractSmoother end

const KS = KalmanSmoother()

struct StateCallback{T}
    proposed_states::Vector{Gaussian{Vector{T},Matrix{T}}}
    filtered_states::Vector{Gaussian{Vector{T},Matrix{T}}}
end
function StateCallback(N::Integer, T::Type)
    return StateCallback{T}(
        Vector{Gaussian{Vector{T},Matrix{T}}}(undef, N),
        Vector{Gaussian{Vector{T},Matrix{T}}}(undef, N),
    )
end

function (callback::StateCallback)(
    model::LinearGaussianStateSpaceModel, algo::KalmanFilter, states, obs; kwargs...
)
    return nothing
end

function (callback::StateCallback)(
    model::LinearGaussianStateSpaceModel,
    algo::KalmanFilter,
    iter::Integer,
    states,
    obs;
    kwargs...,
)
    callback.proposed_states[iter] = states.proposed
    callback.filtered_states[iter] = states.filtered
    return nothing
end

function smooth(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    alg::KalmanSmoother,
    observations::AbstractVector;
    t_smooth=1,
    callback=nothing,
    kwargs...,
) where {T}
    cache = StateCallback(length(observations), T)

    filtered, ll = filter(
        rng, model, KalmanFilter(), observations; callback=cache, kwargs...
    )

    back_state = filtered
    for t in (length(observations) - 1):-1:t_smooth
        back_state = backward(
            rng, model, alg, t, back_state, observations[t]; states_cache=cache, kwargs...
        )
    end

    return back_state, ll
end

function backward(
    rng::AbstractRNG,
    model::LinearGaussianStateSpaceModel{T},
    alg::KalmanSmoother,
    iter::Integer,
    back_state,
    obs;
    states_cache,
    kwargs...,
) where {T}
    μ, Σ = GaussianDistributions.pair(back_state)
    μ_pred, Σ_pred = GaussianDistributions.pair(states_cache.proposed_states[iter + 1])
    μ_filt, Σ_filt = GaussianDistributions.pair(states_cache.filtered_states[iter])

    G = Σ_filt * model.dyn.A' * inv(Σ_pred)
    μ = μ_filt .+ G * (μ .- μ_pred)
    Σ = Σ_filt .+ G * (Σ .- Σ_pred) * G'

    return Gaussian(μ, Σ)
end
