export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model

import SSMProblems: distribution
import Distributions: MvNormal
import LinearAlgebra: cholesky

abstract type LinearGaussianLatentDynamics{T} <: SSMProblems.LatentDynamics{Vector{T}} end

function calc_μ0 end
function calc_Σ0 end
function calc_initial(dyn::LinearGaussianLatentDynamics, extra)
    return calc_μ0(dyn, extra), calc_Σ0(dyn, extra)
end

function calc_A end
function calc_b end
function calc_Q end
function calc_params(dyn::LinearGaussianLatentDynamics, step::Integer, extra)
    return (calc_A(dyn, step, extra), calc_b(dyn, step, extra), calc_Q(dyn, step, extra))
end

abstract type LinearGaussianObservationProcess{T} <:
              SSMProblems.ObservationProcess{Vector{T}} end

function calc_H end
function calc_c end
function calc_R end
function calc_params(obs::LinearGaussianObservationProcess, step::Integer, extra)
    return (
        AnalyticalFilters.calc_H(obs, step, extra),
        calc_c(obs, step, extra),
        calc_R(obs, step, extra),
    )
end

const LinearGaussianStateSpaceModel{T} = SSMProblems.StateSpaceModel{
    T,D,O
} where {T,D<:LinearGaussianLatentDynamics{T},O<:LinearGaussianObservationProcess{T}}

# TODO: this is hacky and should ideally be removed
# Can't use `eltype` because that is used by SSMProblems for forward simulation and would be
# used by a particle filtering.
function rb_eltype(::LinearGaussianStateSpaceModel{T}) where {T}
    return @NamedTuple{μ::Vector{T}, Σ::Matrix{T}} where {T}
end

#######################
#### DISTRIBUTIONS ####
#######################

function SSMProblems.distribution(dyn::LinearGaussianLatentDynamics, extra)
    μ0, Σ0 = calc_initial(dyn, extra)
    return MvNormal(μ0, Σ0)
end

function SSMProblems.distribution(
    dyn::LinearGaussianLatentDynamics{T}, step::Integer, state::AbstractVector{T}, extra
) where {T}
    A, b, Q = calc_params(dyn, step, extra)
    return MvNormal(A * state + b, Q)
end

function SSMProblems.distribution(
    obs::LinearGaussianObservationProcess{T}, step::Integer, state::AbstractVector{T}, extra
) where {T}
    H, c, R = calc_params(obs, step, extra)
    return MvNormal(H * state + c, R)
end

###########################################
#### HOMOGENEOUS LINEAR GAUSSIAN MODEL ####
###########################################

struct HomogeneousLinearGaussianLatentDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    Q::Matrix{T}
end
calc_μ0(dyn::HomogeneousLinearGaussianLatentDynamics, extra) = dyn.μ0
calc_Σ0(dyn::HomogeneousLinearGaussianLatentDynamics, extra) = dyn.Σ0
calc_A(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.A
calc_b(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.b
calc_Q(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.Q

struct HomogeneousLinearGaussianObservationProcess{T} <: LinearGaussianObservationProcess{T}
    H::Matrix{T}
    c::Vector{T}
    R::Matrix{T}
end
calc_H(obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra) = obs.H
calc_c(obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra) = obs.c
calc_R(obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra) = obs.R

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
function batch_calc_initial(dyn::LinearGaussianLatentDynamics, extra, N::Integer)
    return batch_calc_μ0s(dyn, extra, N), batch_calc_Σ0s(dyn, extra, N)
end

function batch_calc_params(
    dyn::LinearGaussianLatentDynamics, step::Integer, extra, N::Integer
)
    return (
        batch_calc_As(dyn, step, extra, N),
        batch_calc_bs(dyn, step, extra, N),
        batch_calc_Qs(dyn, step, extra, N),
    )
end

function batch_calc_params(
    obs::LinearGaussianObservationProcess, step::Integer, extra, N::Integer
)
    return (
        batch_calc_Hs(obs, step, extra, N),
        batch_calc_cs(obs, step, extra, N),
        batch_calc_Rs(obs, step, extra, N),
    )
end

function batch_simulate(dyn::HomogeneousLinearGaussianLatentDynamics, extra, N::Integer)
    μ0, Σ0 = AnalyticalFilters.calc_initial(dyn, extra)
    D = length(μ0)
    L = cholesky(Σ0).L
    # Ls = repeat(cu(reshape(Σ0, (size(Σ0)..., 1))), 1, 1, N)
    Ls = CuArray{Float32}(undef, size(Σ0)..., N)
    Ls[:, :, :] .= cu(Σ0)
    return cu(μ0) .+ NNlib.batched_vec(Ls, CUDA.randn(D, N))
end

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

function batch_calc_Hs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra, N::Integer
)
    H = CuArray{Float32}(undef, size(obs.H)..., N)
    return H[:, :, :] .= cu(obs.H)
end
function batch_calc_cs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra, N::Integer
)
    c = CuArray{Float32}(undef, size(obs.c)..., N)
    return c[:, :] .= cu(obs.c)
end

function batch_calc_Rs(
    obs::HomogeneousLinearGaussianObservationProcess, ::Integer, extra, N::Integer
)
    R = CuArray{Float32}(undef, size(obs.R)..., N)
    return R[:, :, :] .= cu(obs.R)
end
