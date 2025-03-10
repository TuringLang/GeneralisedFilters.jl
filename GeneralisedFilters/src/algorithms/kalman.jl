export KalmanFilter, filter, BatchKalmanFilter
using GaussianDistributions
using CUDA: i32
import LinearAlgebra: Symmetric

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
    S = Symmetric(H * Σ * H' + R)
    K = Σ * H' / S

    filtered = Gaussian(μ + K * y, Σ - K * H * Σ)

    # Compute log-likelihood
    ll = logpdf(MvNormal(m, S), obs)

    return filtered, ll
end

function batch_matmul_nt(A, B)
    matrices_per_block = div(1024, size(A, 1) * size(B, 1))
    shmem = (
        (size(A, 1) * size(A, 2) + size(B, 2) * size(B, 1)) *
        sizeof(eltype(A)) *
        matrices_per_block
    )
    if shmem < attribute(device(), CUDA.DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        # Perform checks
        if size(A, 2) != size(B, 2)
            throw(ArgumentError("Matrix dimensions must agree"))
        end
        if size(A, 3) != size(B, 3)
            throw(ArgumentError("Batch dimensions must agree"))
        end
        if eltype(A) != eltype(B)
            throw(ArgumentError("Matrix types must agree"))
        end
        C = similar(A, size(A, 1), size(B, 1), size(A, 3))
        threads = matrices_per_block * size(A, 1) * size(B, 1)
        blocks = ceil(Int, size(A, 3) / matrices_per_block)
        @cuda blocks = blocks threads = threads gemm_static_shmem_element_kernel!(
            C,
            A,
            B,
            Int32(size(A, 3)),
            Val(Int32(size(A, 1))),
            Val(Int32(size(A, 2))),
            Val(Int32(size(B, 1))),
            Val(Int32(matrices_per_block)),
        )
        return C
    else
        return NNlib.batched_mul(A, NNlib.batched_transpose(B))
    end
end

function gemm_static_shmem_element_kernel!(
    C, A, B, batch_size::Int32, ::Val{M}, ::Val{N}, ::Val{P}, ::Val{K}
) where {M,N,P,K}
    tid = threadIdx().x

    # Loading phase - coalesced reads down columns
    shmem_a = CuStaticSharedArray(Float32, (M, N, K))
    shmem_b = CuStaticSharedArray(Float32, (P, N, K))

    # Load A
    for i in tid:(blockDim().x):(M * N * K)
        matrix = div(i - 1i32, M * N) + 1i32
        mn_idx = mod1(i, M * N)
        m = mod1(mn_idx, M)
        n = div(mn_idx - 1i32, M) + 1i32
        matrix_idx = (blockIdx().x - 1i32) * K + matrix

        if matrix_idx <= batch_size
            shmem_a[m, n, matrix] = A[m, n, matrix_idx]
        end
    end

    # Load B - separate loop with no dependency on M
    for i in tid:(blockDim().x):(P * N * K)
        matrix = div(i - 1i32, P * N) + 1i32
        pn_idx = mod1(i, P * N)
        p = mod1(pn_idx, P)
        n = div(pn_idx - 1i32, P) + 1i32
        matrix_idx = (blockIdx().x - 1i32) * K + matrix

        if matrix_idx <= batch_size
            shmem_b[p, n, matrix] = B[p, n, matrix_idx]
        end
    end

    sync_threads()

    # Computing phase - each thread computes one element of C
    local_matrix = div(tid - 1i32, M * P) + 1i32
    matrix_thread = mod1(tid, M * P)
    row = mod1(matrix_thread, M)
    col = div(matrix_thread - 1i32, M) + 1i32
    matrix_idx = (blockIdx().x - 1i32) * K + local_matrix

    if matrix_idx <= batch_size && row <= M && col <= P
        @inbounds begin
            result = 0.0f0
            for k in 1:N
                result += shmem_a[row, k, local_matrix] * shmem_b[col, k, local_matrix]
            end
            C[row, col, matrix_idx] = result
        end
    end

    return nothing
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
    Σ̂s = batch_matmul_nt(NNlib.batched_mul(As, Σs), As) .+ Qs
    return BatchGaussianDistribution(μ̂s, Σ̂s)
end

function invert_innovation(S)
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
    S = batch_matmul_nt(NNlib.batched_mul(Hs, Σs), Hs) .+ Rs

    ΣH_T = batch_matmul_nt(Σs, Hs)

    S_inv, log_dets = invert_innovation(S)

    K = NNlib.batched_mul(ΣH_T, S_inv)

    μ_filt = μs .+ NNlib.batched_vec(K, y_res)
    Σ_filt = Σs .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs))

    y = cu(obs)

    # TODO: this is y_res
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
