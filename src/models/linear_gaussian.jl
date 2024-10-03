export LinearGaussianLatentDynamics
export LinearGaussianObservationProcess
export LinearGaussianStateSpaceModel
export create_homogeneous_linear_gaussian_model

import SSMProblems: distribution
import Distributions: MvNormal

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

struct HomogeneousLinearGaussianLatentDynamics{
    T,V<:AbstractVector{T},M_ge<:AbstractMatrix{T},M_cov<:AbstractMatrix{T}
} <: LinearGaussianLatentDynamics{T}
    μ0::V
    Σ0::M_cov
    A::M_ge
    b::V
    Q::M_cov
end
calc_μ0(dyn::HomogeneousLinearGaussianLatentDynamics, extra) = dyn.μ0
calc_Σ0(dyn::HomogeneousLinearGaussianLatentDynamics, extra) = dyn.Σ0
calc_A(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.A
calc_b(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.b
calc_Q(dyn::HomogeneousLinearGaussianLatentDynamics, ::Integer, extra) = dyn.Q

struct HomogeneousLinearGaussianObservationProcess{
    T,V<:AbstractVector{T},M_ge<:AbstractMatrix{T},M_cov<:AbstractMatrix{T}
} <: LinearGaussianObservationProcess{T}
    H::M_ge
    c::V
    R::M_cov
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
