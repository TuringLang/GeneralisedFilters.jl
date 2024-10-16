export KalmanFilter, filter, BatchKalmanFilter

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
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    extra,
) where {T}
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
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    obs::Vector{T},
    extra,
) where {T}
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
    ll = logpdf(MvNormal(m, S), obs)

    return (μ=μ̂, Σ=Σ̂), ll
end

function step(
    model::LinearGaussianStateSpaceModel{T},
    filter::KalmanFilter,
    step::Integer,
    state::@NamedTuple{μ::Vector{T}, Σ::Matrix{T}},
    obs::Vector{T},
    extra,
) where {T}
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

struct BatchKalmanFilter <: FilteringAlgorithm
    batch_size::Int
end

function initialise(
    model::LinearGaussianStateSpaceModel{T}, algo::BatchKalmanFilter, extra
) where {T}
    μ0s, Σ0s = batch_calc_initial(model.dyn, extra, algo.batch_size)
    return (μs=μ0s, Σs=Σ0s)
end

function predict(
    model::LinearGaussianStateSpaceModel{T},
    algo::BatchKalmanFilter,
    step::Integer,
    state::@NamedTuple{μs::A1, Σs::A2},
    extra,
) where {T,A1<:CuArray,A2<:CuArray}
    μs, Σs = state.μs, state.Σs
    As, bs, Qs = batch_calc_params(model.dyn, step, extra, algo.batch_size)
    μ̂s = NNlib.batched_vec(As, μs) .+ bs
    Σ̂s = NNlib.batched_mul(NNlib.batched_mul(As, Σs), NNlib.batched_transpose(As)) .+ Qs
    return (μs=μ̂s, Σs=Σ̂s)
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
    S_inv = CuArray{Float32}(undef, size(S))
    # TODO: This fails when D_obs > D_inner since S is not invertible
    CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)

    diags = CuArray{Float32}(undef, size(S, 1), size(S, 3))
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
    state::@NamedTuple{μs::A1, Σs::A2},
    obs::Vector{T},
    extra,
) where {T,A1<:CuArray,A2<:CuArray}
    μs, Σs = state.μs, state.Σs
    Hs, cs, Rs = batch_calc_params(model.obs, step, extra, algo.batch_size)

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
        -0.5f0 * NNlib.batched_vec(reshape(y .- m, 1, size(y, 1), size(S, 3)), inv_term)
    D = size(y, 1)
    log_likes = log_likes .- 0.5f0 * log_dets .- D / 2 * log(T(2π))

    # HACK: only errors seems to be from numerical stability so will just overwrite
    log_likes[isnan.(log_likes)] .= -Inf

    return (μs=μ_filt, Σs=Σ_filt), dropdims(log_likes; dims=1)
end

function step(
    model::LinearGaussianStateSpaceModel{T},
    filter::BatchKalmanFilter,
    step::Integer,
    state::@NamedTuple{μs::A1, Σs::A2},
    obs::Vector{T},
    extra,
) where {T,A1<:CuArray,A2<:CuArray}
    state = predict(model, filter, step, state, extra)
    state, lls = update(model, filter, step, state, obs, extra)
    return state, lls
end
