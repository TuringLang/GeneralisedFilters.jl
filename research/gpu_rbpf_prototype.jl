using TestEnv
TestEnv.activate()

using AnalyticalFilters
using BenchmarkTools
using Distributions
using HypothesisTests
using LinearAlgebra
using LogExpFunctions: softmax
using StableRNGs
using StatsBase

using CUDA
using NNlib

# TODO: add RNGs
# TODO: sort out dims order. NNlibs likes (..., N), CuBLAS likes (N, ...)
# TODO: switched to right division
# TODO: getri_strided_batched! : CUDA has no strided batched getri

# NOTE: it seems like the CuArray APIs make everything colmun major

######################
#### GROUND TRUTH ####
######################

# Define inner dynamics
struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    C::Matrix{T}
    Q::Matrix{T}
end
AnalyticalFilters.calc_μ0(dyn::InnerDynamics, extra) = dyn.μ0
AnalyticalFilters.calc_Σ0(dyn::InnerDynamics, extra) = dyn.Σ0
AnalyticalFilters.calc_A(dyn::InnerDynamics, ::Integer, extra) = dyn.A
function AnalyticalFilters.calc_b(dyn::InnerDynamics, ::Integer, extra)
    return dyn.b + dyn.C * extra.prev_outer
end
AnalyticalFilters.calc_Q(dyn::InnerDynamics, ::Integer, extra) = dyn.Q

rng = StableRNG(1236)
μ0 = rand(rng, 4)
Σ0s = [rand(rng, 2, 2) for _ in 1:2]
Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
Σ0 = [
    Σ0s[1] zeros(2, 2)
    zeros(2, 2) Σ0s[2]
]
A = [
    rand(rng, 2, 2) zeros(2, 2)
    rand(rng, 2, 4)
]
# Make mean-reverting
A /= 3.0
A[diagind(A)] .= -0.5
b = rand(rng, 4)
Qs = [rand(rng, 2, 2) / 10.0 for _ in 1:2]
Qs = [Q * Q' for Q in Qs]  # make Q positive definite
Q = [
    Qs[1] zeros(2, 2)
    zeros(2, 2) Qs[2]
]
H = [zeros(2, 2) rand(rng, 2, 2)]
c = rand(rng, 2)
R = rand(rng, 2, 2)
R = R * R' / 3.0  # make R positive definite

N_particles = 10000
T = 1

observations = [rand(rng, 2) for _ in 1:T]
extras = [nothing for _ in 1:T]

outer_dyn = AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics(
    μ0[1:2], Σ0[1:2, 1:2], A[1:2, 1:2], b[1:2], Qs[1]
)
inner_dyn = InnerDynamics(μ0[3:4], Σ0[3:4, 3:4], A[3:4, 3:4], b[3:4], A[3:4, 1:2], Qs[2])
obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(H[:, 3:4], c, R)
hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

rbpf = RBPF(KalmanFilter(), N_particles, 0.0)
# (xs, zs, log_ws), ll = AnalyticalFilters.filter(rng, hier_model, rbpf, observations, extras)
(xs, zs, log_ws), ll = AnalyticalFilters.filter(
    hier_model, rbpf, observations, nothing, extras
)

println("Finish ground truth")

############################
#### GPU IMPLEMENTATION ####
############################

## Initialisation

function batch_calc_μ0s(dyn::InnerDynamics{T}, N::Integer) where {T}
    μ0s = CuArray{Float32}(undef, length(dyn.μ0), N)
    return μ0s[:, :] .= cu(dyn.μ0)
end

# (..., N)
μ0s = batch_calc_μ0s(inner_dyn, N_particles)

function batch_calc_Σ0s(dyn::InnerDynamics{T}, N::Integer) where {T}
    Σ0s = CuArray{Float32}(undef, size(dyn.Σ0)..., N)
    return Σ0s[:, :, :] .= cu(dyn.Σ0)
end

# (..., N)
Σ0s = batch_calc_Σ0s(inner_dyn, N_particles)

function batch_simulate(
    dyn::AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics, N::Integer
)
    μ0, Σ0 = AnalyticalFilters.calc_initial(dyn, nothing)
    D = length(μ0)
    L = cholesky(Σ0).L
    # Ls = repeat(cu(reshape(Σ0, (size(Σ0)..., 1))), 1, 1, N)
    Ls = CuArray{Float32}(undef, size(Σ0)..., N)
    Ls[:, :, :] .= cu(Σ0)
    return cu(μ0) .+ NNlib.batched_vec(Ls, CUDA.randn(D, N))
end

# (..., N)
x0s = batch_simulate(outer_dyn, N_particles);

logws = CUDA.fill(convert(Float32, -log(N_particles)), N_particles);

## Prediction

function batch_simulate(
    dyn::AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics,
    step::Integer,
    prev_state,
    extra,
    N::Integer,
)
    A, b, Q = AnalyticalFilters.calc_params(dyn, step, extra)
    D = length(b)
    L = cholesky(Q).L
    Ls = CuArray{Float32}(undef, size(Q)..., N)
    Ls[:, :, :] .= cu(Q)
    As = CuArray{Float32}(undef, size(A)..., N)
    As[:, :, :] .= cu(A)
    return (NNlib.batched_vec(As, prev_state) .+ cu(b)) +
           NNlib.batched_vec(Ls, CUDA.randn(D, N))
end

x1s = batch_simulate(outer_dyn, 1, x0s, nothing, N_particles);

function batch_calc_A(dyn::InnerDynamics, ::Integer, extra, N::Integer)
    As = CuArray{Float32}(undef, size(dyn.A)..., N)
    As[:, :, :] .= cu(dyn.A)
    return As
end
function batch_calc_b(dyn::InnerDynamics, ::Integer, extra, N::Integer)
    # return dyn.b + dyn.C * extra.prev_outer
    Cs = CuArray{Float32}(undef, size(dyn.C)..., N)
    Cs[:, :, :] .= cu(dyn.C)
    return NNlib.batched_vec(Cs, extra.prev_outer) .+ cu(dyn.b)
end
function batch_calc_Q(dyn::InnerDynamics, ::Integer, extra, N::Integer)
    Q = CuArray{Float32}(undef, size(dyn.Q)..., N)
    return Q[:, :, :] .= cu(dyn.Q)
end

extra = (prev_outer=x0s, new_outer=x1s)

# (..., N)
As = batch_calc_A(inner_dyn, 1, extra, N_particles)
# (..., N)
bs = batch_calc_b(inner_dyn, 1, extra, N_particles)
bs[:, 1]
inner_dyn.b + inner_dyn.C * Array(extra.prev_outer[:, 1])
# (..., N)
Qs = batch_calc_Q(inner_dyn, 1, extra, N_particles)

μs_pred = NNlib.batched_vec(As, μ0s) .+ bs
Σs_pred = NNlib.batched_mul(NNlib.batched_mul(As, Σ0s), NNlib.batched_transpose(As)) .+ Qs

## Update

function batch_calc_H(obs::LinearGaussianObservationProcess, ::Integer, extra, N::Integer)
    H = CuArray{Float32}(undef, size(obs.H)..., N)
    return H[:, :, :] .= cu(obs.H)
end
function batch_calc_c(obs::LinearGaussianObservationProcess, ::Integer, extra, N::Integer)
    c = CuArray{Float32}(undef, size(obs.c)..., N)
    return c[:, :] .= cu(obs.c)
end
function batch_calc_R(obs::LinearGaussianObservationProcess, ::Integer, extra, N::Integer)
    R = CuArray{Float32}(undef, size(obs.R)..., N)
    return R[:, :, :] .= cu(obs.R)
end

Hs = batch_calc_H(obs, 1, nothing, N_particles)
cs = batch_calc_c(obs, 1, nothing, N_particles)
Rs = batch_calc_R(obs, 1, nothing, N_particles)

m = NNlib.batched_vec(Hs, μs_pred) .+ cs
y = cu(observations[1]) .- m
S = NNlib.batched_mul(NNlib.batched_mul(Hs, Σs_pred), NNlib.batched_transpose(Hs)) .+ Rs

S[:, :, 1]
Array(Hs[:, :, 1]) * Array(Σs_pred[:, :, 1]) * Array(Hs[:, :, 1])' .+ Array(Rs[:, :, 1])

ΣH_T = NNlib.batched_mul(Σs_pred, NNlib.batched_transpose(Hs))
HΣ = NNlib.batched_mul(Hs, Σs_pred)

d_ipiv, _, d_S = CUDA.CUBLAS.getrf_strided_batched(S, true)

# TODO: try to find a non-transpose version
_, K_s = CUDA.CUBLAS.getrs_strided_batched('N', d_S, HΣ, d_ipiv)
K = NNlib.batched_transpose(K_s)

S_inv = CuArray{Float32}(undef, 2, 2, N_particles)
CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)
# K = NNlib.batched_mul(ΣH_T, S_inv)

μ_filt = μs_pred .+ NNlib.batched_vec(K, y)
Σ_filt = Σs_pred .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs_pred))

Σ1 = Array(Σs_pred[:, :, 1]);
H1 = Array(Hs[:, :, 1]);
S1 = Array(S[:, :, 1]);
K1 = Σ1 * H1' / S1
K[:, :, 1]
# # Write using inv(S1)
# println(Σ1 * H1' * inv(S1))

# Compute log-likelihood
diags = CuArray{Float32}(undef, 2, N_particles)
diags[1, :] .= d_S[1, 1, :]
diags[2, :] .= d_S[2, 2, :]

log_dets = sum(log, diags; dims=1)
logdet(Array(S[:, :, 1]))

o = cu(observations[1])
_, inv_term = CUDA.CUBLAS.getrs_strided_batched(
    'N', d_S, reshape(o .- m, (2, 1, N_particles)), d_ipiv
)
inv_term = dropdims(inv_term; dims=2)
log_likes = -0.5f0 * NNlib.batched_vec(reshape(o .- m, 1, 2, N_particles), inv_term)

log_likes = log_likes .- 0.5f0 * log_dets .- convert(Float32, log(2π))

m1 = Vector(m[:, 1])
S1 = Array(S[:, :, 1])
S1 = (S1 + S1') / 2
logpdf(MvNormal(m1, S1), observations[1]) - only(Array(log_likes[:, 1]))

println("Finish GPU")

## Comparison

# Ground truth weighted inner mean
weights = softmax(log_ws);
println(sum(getfield.(zs, :μ) .* weights))

# GPU weighted inner mean
weights = softmax(log_likes)
println(convert.(Float64, Array(dropdims(sum(μ_filt .* weights; dims=2); dims=2))))

# Without weights
# println(mean(getfield.(zs, :μ)))
# println(mean(μ_filt; dims=2))

println()

# Ground truth weighted outer mean
weights = softmax(log_ws);
println(sum(xs .* weights))

# GPU weighted outer mean
weights = softmax(log_likes)
println(convert.(Float64, Array(dropdims(sum(x1s .* weights; dims=2); dims=2))))

# Without weights
# println(mean(xs))
# println(mean(x1s; dims=2))

# # Ground truth averacge of covariances
# println(mean(getfield.(zs, :Σ)))

# # GPU average of covariances
# println(mean(Σ_filt; dims=3))

## Component Benchmarking

# # Not present in CPU version
# @benchmark CUDA.@sync batch_calc_μ0s(inner_dyn, N_particles)

# # Not present in CPU version
# @benchmark CUDA.@sync batch_calc_Σ0s(inner_dyn, N_particles)

# # TODO: Both of these use the inefficient `repeat`
# @benchmark CUDA.@sync batch_simulate(outer_dyn, N_particles)
# @benchmark CUDA.@sync batch_simulate(outer_dyn, 1, x0s, nothing, N_particles)

# # Not present in CPU version
# @benchmark CUDA.@sync batch_calc_A(inner_dyn, 1, extra, N_particles)

# @benchmark CUDA.@sync batch_calc_b(inner_dyn, 1, extra, N_particles)

# # Not present in CPU version
# @benchmark CUDA.@sync batch_calc_Q(inner_dyn, 1, extra, N_particles)

# # This is crazy amounts faster than the other functions
# @benchmark CUDA.@sync NNlib.batched_vec(As, μ0s) .+ bs

# # Maybe we can still make it faster by avoiding allocations
# function fast_batch_Ab_plus_c(As, bs, cs)
#     cs_copy = copy(cs)
#     CUDA.CUBLAS.gemv_strided_batched!('N', 1.0f0, As, bs, 1.0f0, cs_copy)
#     return cs_copy
# end

# NNlib.batched_vec(As, μ0s) .+ bs
# fast_batch_Ab_plus_c(As, μ0s, bs)

# # Still 500. Allocations don't seem to be a big deal
# @benchmark CUDA.@sync fast_batch_Ab_plus_c(As, μ0s, bs)
# @benchmark CUDA.@sync CuArray{Float32}(undef, 2, N_particles)  # yeah...basically 1μs

# # Attempt to speed up
# # function batch_calc_A(dyn::InnerDynamics, ::Integer, extra, N::Integer)
# #     A = cu(reshape(dyn.A, (size(dyn.A)..., 1)))
# #     As = repeat(A, 1, 1, N)
# #     return As
# # end

# @benchmark CUDA.@sync batch_calc_A(inner_dyn, 1, extra, N_particles)

# function batch_calc_A_1(dyn::InnerDynamics, ::Integer, extra, N::Integer)
#     A = CuArray{Float32}(undef, 2, 2, N)
#     return A[:, :, :] .= cu(dyn.A)
# end
# batch_calc_A_1(inner_dyn, 1, extra, N_particles)

# @benchmark CUDA.@sync batch_calc_A_1(inner_dyn, 1, extra, N_particles)

# # 3000x speedup! Looks like most of the time was spent on this

# # TODO: find out why minimum is so much less than the median...CUDA.@sync seems to fix it

# # A lot more than the calculation of μs_pred!
# @benchmark CUDA.@sync NNlib.batched_mul(
#     NNlib.batched_mul(As, Σ0s), NNlib.batched_transpose(As)
# ) .+ Qs

# @benchmark CUDA.@sync NNlib.batched_transpose(As)
# @benchmark CUDA.@sync NNlib.batched_mul(As, Σ0s)
# @benchmark CUDA.@sync NNlib.batched_mul(
#     NNlib.batched_mul(As, Σ0s), NNlib.batched_transpose(As)
# )

function transpose_first(As, Bs)
    Bs_new = CuArray{Float32}(undef, size(Bs))
    Bs_new[1, 1, :] .= Bs[1, 1, :]
    Bs_new[1, 2, :] .= Bs[2, 1, :]
    Bs_new[2, 1, :] .= Bs[1, 2, :]
    Bs_new[2, 2, :] .= Bs[2, 2, :]
    return NNlib.batched_mul(As, Bs_new)
end

# @benchmark CUDA.@sync transpose_first(NNlib.batched_mul(As, Σ0s), As)

# # Custom kernel for batched transposing
# function batch_transpose(As)
#     out = CuArray{Float32}(undef, size(As))
#     @cuda threads = 256 blocks = div(length(As), 256) _batch_transpose!(As, out)
#     return out
# end

# function _batch_transpose!(As, out)
#     index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
#     stride = gridDim().x * blockDim().x
#     for i in index:stride:size(As, 3)
#         for j in 1:size(As, 2)
#             for k in 1:size(As, 1)
#                 @inbounds out[k, j, i] = As[j, k, i]
#             end
#         end
#     end
#     return nothing
# end

# function batch_transpose_thread_optim(As)
#     out = CuArray{Float32}(undef, size(As))
#     kernel = @cuda launch = false _batch_transpose!(As, out)
#     config = launch_configuration(kernel.fun)
#     threads = min(size(As, 3), config.threads)
#     blocks = cld(size(As, 3), threads)
#     @cuda threads = threads blocks = blocks _batch_transpose!(As, out)
#     return out
# end

# @benchmark CUDA.@sync CuArray(NNlib.batched_transpose(As))
# @benchmark CUDA.@sync batch_transpose(As)
# @benchmark CUDA.@sync batch_transpose_thread_optim(As)

# # Seems to call a slower version when transpose present even though transpose itself is fast
# @benchmark CUDA.@sync NNlib.batched_mul(NNlib.batched_mul(As, Σ0s), As)

# # Weird...still present when using gemm. Is it a memory contiguity issue?
# @benchmark CUDA.@sync CUDA.CUBLAS.gemm_strided_batched(
#     'N', 'T', 1.0f0, NNlib.batched_mul(As, Σ0s), As
# )

# # MORE INEFFICIENT CALC FUNCTIONS...

# @benchmark CUDA.@sync NNlib.batched_vec(Hs, μs_pred) .+ cs

# @benchmark CUDA.@sync NNlib.batched_mul(
#     NNlib.batched_mul(Hs, Σs_pred), NNlib.batched_transpose(Hs)
# ) .+ Rs

# # biggest things so far 17ms * 2 = 34ms.  Could potentially be made faster if figure out how
# # to avoid the transpose slow down

# @benchmark CUDA.@sync NNlib.batched_mul(Σs_pred, NNlib.batched_transpose(Hs))

# # This is expensive at 200ms. Could improve by using Cholesky (roughly 2x faster)
# # Surprised it takes so long for such a small matrix. Doesn't feel that much harder than
# # gemv
# # Same speed as TensorFlow using Cholesky. Weirdly LU was faster (strided?) at 170ms
# # Worse, CPU is 237ms
# # Code on Germain is far faster (though perhaps not as numerically stable). Roughly 7ms
# @benchmark CUDA.@sync CUDA.CUBLAS.getrf_strided_batched(S, true)

# # What about a super fast implementation for 2x2 matrices?
# function fast_cholesky(S)
#     out = CUDA.zeros(size(S))
#     out[1, 1, :] .= sqrt.(S[1, 1, :])
#     out[2, 1, :] .= S[2, 1, :] ./ out[1, 1, :]
#     out[2, 2, :] .= sqrt.(S[2, 2, :] .- out[2, 1, :] .^ 2)
#     return out
# end
# S1 = Array(S[:, :, 1])
# S1 = (S1 + S1') / 2
# cholesky(S1).L - Array(fast_cholesky(S)[:, :, 1])

# # 74 μs !!!
# @benchmark CUDA.@sync fast_cholesky(S)

# # Even slower–450ms but can be avoided at least
# @benchmark CUDA.@sync CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)

# @benchmark CUDA.@sync NNlib.batched_mul(ΣH_T, S_inv)

# # These will also be quick
# # μ_filt = μs_pred .+ NNlib.batched_vec(K, y)
# # Σ_filt = Σs_pred .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs_pred))

# @benchmark CUDA.@sync sum(log, [d_S[1, 1, :]; d_S[2, 2, :]]; dims=1)

# # Note, got around transpose by reshaping
# @benchmark CUDA.@sync -0.5f0 * NNlib.batched_vec(
#     reshape(o .- m, 1, 2, N_particles), NNlib.batched_vec(S_inv, o .- m)
# )

function filter(
    model::HierarchicalSSM, N_particles::Integer, observations::Vector{Vector{Float64}}
)
    inner_dyn = model.inner_model.dyn
    outer_dyn = model.outer_dyn
    obs = model.inner_model.obs

    μ0s = batch_calc_μ0s(inner_dyn, N_particles)
    Σ0s = batch_calc_Σ0s(inner_dyn, N_particles)
    x0s = batch_simulate(outer_dyn, N_particles)
    logws = CUDA.fill(convert(Float32, -log(N_particles)), N_particles)

    x1s = batch_simulate(outer_dyn, 1, x0s, nothing, N_particles)
    extra = (prev_outer=x0s, new_outer=x1s)

    As = batch_calc_A(inner_dyn, 1, extra, N_particles)
    bs = batch_calc_b(inner_dyn, 1, extra, N_particles)
    Qs = batch_calc_Q(inner_dyn, 1, extra, N_particles)

    μs_pred = NNlib.batched_vec(As, μ0s) .+ bs
    Σs_pred =
        NNlib.batched_mul(NNlib.batched_mul(As, Σ0s), NNlib.batched_transpose(As)) .+ Qs
    # Σs_pred = transpose_first(NNlib.batched_mul(As, Σ0s), As) .+ Qs

    Hs = batch_calc_H(obs, 1, nothing, N_particles)
    cs = batch_calc_c(obs, 1, nothing, N_particles)
    Rs = batch_calc_R(obs, 1, nothing, N_particles)

    m = NNlib.batched_vec(Hs, μs_pred) .+ cs
    y = cu(observations[1]) .- m
    S = NNlib.batched_mul(NNlib.batched_mul(Hs, Σs_pred), NNlib.batched_transpose(Hs)) .+ Rs
    # S = transpose_first(NNlib.batched_mul(Hs, Σs_pred), Hs) .+ Rs

    ΣH_T = NNlib.batched_mul(Σs_pred, NNlib.batched_transpose(Hs))
    # ΣH_T = transpose_first(Σs_pred, Hs)

    d_ipiv, _, d_S = CUDA.CUBLAS.getrf_strided_batched(S, true)

    # _, K_s = CUDA.CUBLAS.getrs_strided_batched('N', d_S, ΣH_T, d_ipiv)
    # K = NNlib.batched_transpose(K_s)

    S_inv = CuArray{Float32}(undef, 2, 2, N_particles)
    CUDA.CUBLAS.getri_strided_batched!(d_S, S_inv, d_ipiv)
    K = NNlib.batched_mul(ΣH_T, S_inv)

    μ_filt = μs_pred .+ NNlib.batched_vec(K, y)
    Σ_filt = Σs_pred .- NNlib.batched_mul(K, NNlib.batched_mul(Hs, Σs_pred))

    o = cu(observations[1])
    log_dets = sum(log, [d_S[1, 1, :]; d_S[2, 2, :]]; dims=1)
    # _, inv_term = CUDA.CUBLAS.getrs_strided_batched(
    #     'N', d_S, reshape(o .- m, (2, 1, N_particles)), d_ipiv
    # )
    # inv_term = dropdims(inv_term; dims=2)
    inv_term = NNlib.batched_vec(S_inv, o .- m)
    log_likes = -0.5f0 * NNlib.batched_vec(reshape(o .- m, 1, 2, N_particles), inv_term)
    log_likes = log_likes .- 0.5f0 * log_dets .- convert(Float32, log(2π))

    return μ_filt, Σ_filt, log_likes
end

## Combined Benchmarking

io = IOContext(stdout)

res_cpu = @benchmark AnalyticalFilters.filter(
    hier_model, rbpf, observations, nothing, extras
)
show(io, MIME("text/plain"), res_cpu)
res_gpu = @benchmark CUDA.@sync filter(hier_model, N_particles, observations)
show(io, MIME("text/plain"), res_gpu)
