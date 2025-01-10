export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model

import SSMProblems: distribution
import Distributions: MvNormal
import LinearAlgebra: cholesky

abstract type LinearGaussianLatentDynamics{T} <: SSMProblems.LatentDynamics{T,Vector{T}} end

function calc_μ0 end
function calc_Σ0 end
function calc_initial(dyn::LinearGaussianLatentDynamics; kwargs...)
    return calc_μ0(dyn; kwargs...), calc_Σ0(dyn; kwargs...)
end

function calc_A end
function calc_b end
function calc_Q end
function calc_params(dyn::LinearGaussianLatentDynamics, step::Integer; kwargs...)
    return (
        calc_A(dyn, step; kwargs...),
        calc_b(dyn, step; kwargs...),
        calc_Q(dyn, step; kwargs...),
    )
end

abstract type LinearGaussianObservationProcess{T} <:
              SSMProblems.ObservationProcess{T,Vector{T}} end

function calc_H end
function calc_c end
function calc_R end
function calc_params(obs::LinearGaussianObservationProcess, step::Integer; kwargs...)
    return (
        GeneralisedFilters.calc_H(obs, step; kwargs...),
        calc_c(obs, step; kwargs...),
        calc_R(obs, step; kwargs...),
    )
end

const LinearGaussianStateSpaceModel{T} = SSMProblems.StateSpaceModel{
    T,D,O
} where {T,D<:LinearGaussianLatentDynamics{T},O<:LinearGaussianObservationProcess{T}}

function rb_eltype(::LinearGaussianStateSpaceModel{T}) where {T}
    return Gaussian{Vector{T},Matrix{T}}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(dyn::LinearGaussianLatentDynamics; kwargs...)
    μ0, Σ0 = calc_initial(dyn; kwargs...)
    return MvNormal(μ0, Σ0)
end

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics, step::Integer, state::AbstractVector; kwargs...
)
    A, b, Q = calc_params(dyn, step; kwargs...)
    return MvNormal(A * state + b, Q)
end

function SSMProblems.distribution(
    obs::LinearGaussianObservationProcess, step::Integer, state::AbstractVector; kwargs...
)
    H, c, R = calc_params(obs, step; kwargs...)
    return MvNormal(H * state + c, R)
end

##########################
#### BATCH SIMULATION ####
##########################

function cholesky_from_lu(Ls_lu::CuArray{T}, D, N) where {T}
    # Compute Cholesky factor from L (scale L/U to have equal diagonals)
    # This is not a particularly optimised approach since it will be replaced later
    diags = CUDA.zeros(T, D, D, N)
    for d in 1:D
        diags[d, d, :] .= Ls_lu[d, d, :]
    end
    Ls = CUDA.zeros(T, D, D, N)
    for d in 1:D
        Ls[d, d, :] .= 1.0
    end
    for i in 1:D
        for j in 1:(i - 1)
            Ls[i, j, :] .= Ls_lu[i, j, :]
        end
    end
    return NNlib.batched_mul(Ls, sqrt.(diags))
end

function SSMProblems.batch_simulate(
    ::AbstractRNG, dyn::LinearGaussianLatentDynamics{T}, N::Integer; kwargs...
) where {T}
    μ0s, Σ0s = batch_calc_initial(dyn, N; kwargs...)
    D = size(μ0s, 1)
    # Compute Cholesky factor using LU decomposition (no pivoting needed for Cholesky)
    # TODO: replace this with MAGMA's batched Choleksy
    Ls_lu = CUDA.CUBLAS.getrf_batched(Σ0s, false)
    Ls = cholesky_from_lu(Ls_lu, D, N)
    return μ0s .+ NNlib.batched_vec(Ls, CUDA.randn(T, D, N))
end

function SSMProblems.batch_simulate(
    ::AbstractRNG,
    dyn::LinearGaussianLatentDynamics{T},
    step::Integer,
    prev_state;
    kwargs...,
) where {T}
    N = size(prev_state, 2)
    As, bs, Qs = batch_calc_params(dyn, step, N; kwargs...)
    D = size(prev_state, 1)
    # Compute Cholesky factor using LU decomposition (no pivoting needed for Cholesky)
    Ls_lu = CUDA.CUBLAS.getrf_batched(Qs, false)
    Ls = cholesky_from_lu(Ls_lu, D, size(prev_state, 2))
    return (
        (NNlib.batched_vec(As, prev_state) .+ bs) +
        NNlib.batched_vec(Ls, CUDA.randn(T, D, N))
    )
end

function SSMProblems.batch_simulate(
    ::AbstractRNG, obs::LinearGaussianObservationProcess{T}, step::Integer, state; kwargs...
) where {T}
    N = size(state, 2)
    Hs, cs, Rs = batch_calc_params(obs, step, N; kwargs...)
    D = size(Hs, 1)
    # Compute Cholesky factor using LU decomposition (no pivoting needed for Cholesky)
    Ls_lu = CUDA.CUBLAS.getrf_batched(Rs, false)
    Ls = cholesky_from_lu(Ls_lu, D, N)
    return (NNlib.batched_vec(Hs, state) .+ cs) + NNlib.batched_vec(Ls, CUDA.randn(T, D, N))
end

###########################################
#### HOMOGENEOUS LINEAR GAUSSIAN MODEL ####
###########################################

struct HomogeneousLinearGaussianLatentDynamics{
    T<:Real,ΣT<:AbstractMatrix{T},AT<:AbstractMatrix{T},QT<:AbstractMatrix{T}
} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::ΣT
    A::AT
    b::Vector{T}
    Q::QT
end
calc_μ0(dyn::HomogeneousLinearGaussianLatentDynamics; kwargs...) = dyn.μ0
calc_Σ0(dyn::HomogeneousLinearGaussianLatentDynamics; kwargs...) = dyn.Σ0
calc_A(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.A
calc_b(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.b
calc_Q(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer; kwargs...) = dyn.Q

struct HomogeneousLinearGaussianObservationProcess{
    T<:Real,HT<:AbstractMatrix{T},RT<:AbstractMatrix{T}
} <: LinearGaussianObservationProcess{T}
    H::HT
    c::Vector{T}
    R::RT
end
calc_H(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.H
calc_c(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.c
calc_R(obs::HomogeneousLinearGaussianObservationProcess, ::Integer; kwargs...) = obs.R

function create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    return SSMProblems.StateSpaceModel(
        HomogeneousLinearGaussianLatentDynamics(μ0, Σ0, A, b, Q),
        HomogeneousLinearGaussianObservationProcess(H, c, R),
    )
end

#######################
#### BATCH METHODS ####
#######################

function batch_calc_μ0s end
function batch_calc_Σ0s end
function batch_calc_As end
function batch_calc_bs end
function batch_calc_Qs end
function batch_calc_Hs end
function batch_calc_cs end
function batch_calc_Rs end

# TODO: can we remove batch size argument?
function batch_calc_initial(dyn::LinearGaussianLatentDynamics, N::Integer; kwargs...)
    return batch_calc_μ0s(dyn, N; kwargs...), batch_calc_Σ0s(dyn, N; kwargs...)
end

function batch_calc_params(
    dyn::LinearGaussianLatentDynamics, step::Integer, N::Integer; kwargs...
)
    return (
        batch_calc_As(dyn, step, N; kwargs...),
        batch_calc_bs(dyn, step, N; kwargs...),
        batch_calc_Qs(dyn, step, N; kwargs...),
    )
end

function batch_calc_params(
    obs::LinearGaussianObservationProcess, step::Integer, N::Integer; kwargs...
)
    return (
        batch_calc_Hs(obs, step, N; kwargs...),
        batch_calc_cs(obs, step, N; kwargs...),
        batch_calc_Rs(obs, step, N; kwargs...),
    )
end

function SSMProblems.batch_simulate(
    ::AbstractRNG, dyn::HomogeneousLinearGaussianLatentDynamics{T}, N::Integer; kwargs...
) where {T}
    μ0, Σ0 = GeneralisedFilters.calc_initial(dyn; kwargs...)
    D = length(μ0)
    L = cholesky(Σ0).L
    Ls = CuArray{T}(undef, size(Σ0)..., N)
    Ls[:, :, :] .= cu(L)
    return cu(μ0) .+ NNlib.batched_vec(Ls, CUDA.randn(T, D, N))
end

function SSMProblems.batch_simulate(
    ::AbstractRNG,
    dyn::GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics{T},
    step::Integer,
    prev_state;
    kwargs...,
) where {T}
    N = size(prev_state, 2)
    A, b, Q = GeneralisedFilters.calc_params(dyn, step; kwargs...)
    D = length(b)
    L = cholesky(Q).L
    Ls = CuArray{T}(undef, size(Q)..., N)
    Ls[:, :, :] .= cu(L)
    As = CuArray{T}(undef, size(A)..., N)
    As[:, :, :] .= cu(A)
    return (NNlib.batched_vec(As, prev_state) .+ cu(b)) +
           NNlib.batched_vec(Ls, CUDA.randn(T, D, N))
end

function batch_calc_μ0s(
    dyn::HomogeneousLinearGaussianLatentDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    μ0 = CuArray{T}(undef, length(dyn.μ0), N)
    return μ0[:, :] .= cu(dyn.μ0)
end
function batch_calc_Σ0s(
    dyn::HomogeneousLinearGaussianLatentDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    Σ0 = CuArray{T}(undef, size(dyn.Σ0)..., N)
    return Σ0[:, :, :] .= cu(dyn.Σ0)
end
function batch_calc_As(
    dyn::HomogeneousLinearGaussianLatentDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    A = CuArray{T}(undef, size(dyn.A)..., N)
    return A[:, :, :] .= cu(dyn.A)
end
function batch_calc_bs(
    dyn::HomogeneousLinearGaussianLatentDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    b = CuArray{T}(undef, size(dyn.b)..., N)
    return b[:, :] .= cu(dyn.b)
end
function batch_calc_Qs(
    dyn::HomogeneousLinearGaussianLatentDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    Q = CuArray{T}(undef, size(dyn.Q)..., N)
    return Q[:, :, :] .= cu(dyn.Q)
end

function batch_calc_Hs(
    obs::HomogeneousLinearGaussianObservationProcess{T}, ::Integer, N::Integer; kwargs...
) where {T}
    H = CuArray{T}(undef, size(obs.H)..., N)
    return H[:, :, :] .= cu(obs.H)
end
function batch_calc_cs(
    obs::HomogeneousLinearGaussianObservationProcess{T}, ::Integer, N::Integer; kwargs...
) where {T}
    c = CuArray{T}(undef, size(obs.c)..., N)
    return c[:, :] .= cu(obs.c)
end

function batch_calc_Rs(
    obs::HomogeneousLinearGaussianObservationProcess{T}, ::Integer, N::Integer; kwargs...
) where {T}
    R = CuArray{T}(undef, size(obs.R)..., N)
    return R[:, :, :] .= cu(obs.R)
end
