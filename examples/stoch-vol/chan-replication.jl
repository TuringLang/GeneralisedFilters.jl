using AnalyticalFilters, SSMProblems
using Distributions, Random
using LinearAlgebra, MatrixEquations, Polynomials
using StatsFuns, StatsBase

## ARMA PROCESS ###############################################################

# this is a simplified version of the interface in ARMAModels.jl
struct ARMA{T, p, q} <: LatentDynamics{T}
    φ::Vector{T}
    θ::Vector{T}

    function ARMA(φ::Vector{T}, θ::Vector{T}) where {T<:Real}
        # check for invertibility
        ma_poly = Polynomial([1; θ])
        ma_roots = inv.(roots(ma_poly))
        @assert all(abs2.(ma_roots) .< 1)

        # check for stationarity
        ar_poly = Polynomial([1; -φ])
        ar_roots = roots(ar_poly)
        @assert all(abs2.(ar_roots) .> 1)

        return new{T, length(φ), length(θ)}(φ, θ)
    end
end

function polynomials(proc::ARMA{T, p, q}) where {T<:Real, p, q}
    return (
        φ = Polynomial([1; -proc.φ]),
        θ = Polynomial([1; proc.θ])
    )
end

function Polynomials.roots(proc::ARMA{T, p, q}) where {T<:Real, p, q}
    polys = polynomials(proc)
    return (
        φ = roots(polys.φ),
        θ = inv.(roots(polys.θ))
    )
end

function Base.show(io::IO, proc::ARMA{T, p, q}) where {T<:Real, p, q}
    φ = round.(proc.φ, digits=3)
    θ = round.(proc.θ, digits=3)
    print(io, "ARMA($p,$q){$T}:\n  φ: $φ\n  θ: $θ")
end

# defined alias for autoregressive models AR(p) = ARMA(p,0)
const AR{T, p} = ARMA{T, p, 0}

function AR(φ::Vector{T}) where {T<:Real}
    return ARMA(φ, T[])
end

function Base.show(io::IO, proc::AR{T, p}) where {T<:Real, p}
    φ = round.(proc.φ, digits=3)
    print(io, "AR($p){$T}:\n  φ: $φ")
end

# define alias for moving-average models such that MA(q) = ARMA(0,q)
const MA{T, q} = ARMA{T, 0, q}

function MA(θ::Vector{T}) where {T<:Real}
    return ARMA(T[], θ)
end

function Base.show(io::IO, proc::MA{T, q}) where {T<:Real, q}
    θ = round.(proc.θ, digits=3)
    print(io, "MA($q){$T}:\n  θ: $θ")
end

## STOCHASTIC VOLATILITY PROCESS ##############################################

function stochastic_volatility(γ::T; dims::Integer=2) where {T<:Real}
    return AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics(
        zeros(T, dims),
        Matrix{T}(100I, dims, dims),
        Matrix{T}(I, dims, dims),
        zeros(T, dims),
        Matrix(γ*I, dims, dims)
    )
end

## CHAN'S UNOBSERVED COMPONENTS MODEL #########################################

struct AutoregressiveDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}

    A::Matrix{T}
    b::Vector{T}
    σ²::T

    function AutoregressiveDynamics(ar_poly::Vector{T}, σ²::T, d::Int) where {T<:Real}        
        A1 = vcat(ar_poly[1:d]', diagm(d-1, d, ones(d-1)))
        Q1 = diagm(d, d, T[100])

        return new{T}(
            zeros(T, d+1),
            cat(σ², lyapd(A1, Q1), dims=(1,2)),
            cat(T[1], A1, dims=(1,2)),
            zeros(T, d+1),
            σ²
        )
    end
end

# everything remains constant except volatility
AnalyticalFilters.calc_μ0(dyn::AutoregressiveDynamics, extra0) = dyn.μ0
AnalyticalFilters.calc_Σ0(dyn::AutoregressiveDynamics, extra0) = dyn.Σ0
AnalyticalFilters.calc_A(dyn::AutoregressiveDynamics, ::Integer, extra) = dyn.A
AnalyticalFilters.calc_b(dyn::AutoregressiveDynamics, ::Integer, extra) = dyn.b

# first state is trend variance, second is the error dynamics
function AnalyticalFilters.calc_Q(dyn::AutoregressiveDynamics{T}, ::Integer, extra) where {T<:Real}
    d = size(dyn.A, 1)
    return diagm(d, d, T[dyn.σ²; exp.(extra.new_outer[1])])
end

# see Hamilton's form of the ARMA state space model for reference
function UC(proc::ARMA{T, p, q}, σ²::T, γ::T) where {T<:Real, p, q}
    outer_dyn = stochastic_volatility(γ, dims=1)

    d = max(p, q+1)
    θ = cat(proc.θ, zeros(T, d-1-q), dims=1)
    φ = cat(proc.φ, zeros(T, d-p), dims=1)

    # autoregressive dynamics
    inner_dyn = AutoregressiveDynamics(φ, σ², d)

    # moving average observation process
    obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(
        T[1 1 θ[1:d-1]'...],
        zeros(T, 1),
        T[1e-8;;]
    )

    return HierarchicalSSM(outer_dyn, inner_dyn, obs)
end

function UC(σ²::T, γ::T) where {T<:Real}
    outer_dyn = stochastic_volatility(γ, dims=1)
    inner_dyn = AutoregressiveDynamics(zeros(T, 1), σ², 1)
    obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(
        T[1 1],
        zeros(T, 1),
        T[1e-8;;]
    )

    return HierarchicalSSM(outer_dyn, inner_dyn, obs)
end

## REPLICATION ################################################################

using CSV, DataFrames
using Dates, FredData
using CairoMakie

# I usually use `cca`, but it requires some tinkering with FredData.jl
fred_data = get_data(
    Fred(),
    "PCECTPI",
    observation_start = "1947-04-01",
    observation_end   = "2011-10-01",
    units = "pca"
)

# use PCE instead since quarterly CPI doesn't go back far enough for that
pce_data  = fred_data.data.value
pce_dates = fred_data.data.date

observations = [[pce] for pce in pce_data]

# define basic callback to collect mean and covariance
function filtered_stats(
        rng, model, algo, state, i; iter_array, kwargs...
    )
    xs, zs, logws = state
    μ, Σ = AnalyticalFilters.rb_mean_and_cov(xs, zs, logws)
    iter_array.μ[i,:] = μ
    iter_array.Σ[i,:,:] = Σ
end

extra0 = nothing
extras = [nothing for _ in 1:length(observations)]

rng = MersenneTwister(1234)
rbpf = RBPF(KalmanFilter(), 256, 0.5)
ucma_model = UC(MA([0.463]), 0.141^2, 0.224^2)

# preallocate filtered statistics
filt_stats = (
    μ = zeros(length(observations), 4),
    Σ = zeros(length(observations), 4, 4)
)

# filter with callback
(xs, zs, log_ws), ll = AnalyticalFilters.filter(
    rng, ucma_model, rbpf, observations, extra0, extras;
    callback = filtered_stats, iter_array = filt_stats
)

# format dates for Makie plots
date_format(dates) = x -> [
    Dates.format(dates[floor(Int, i)+1], "yyyy") for i in x
]

# plot the filtered mean using the callback array
ucma_plots = begin
    hist_ts = filt_stats.μ
    fig = Figure()

    ax1 = Axis(
        fig[1, 1],
        title = "trend",
        xtickformat = date_format(pce_dates)
    )
    ax2 = Axis(
        fig[2, 1],
        title = "volatility",
        xtickformat = date_format(pce_dates)
    )

    # exclude the initial draw
    lines!(ax1, hist_ts[2:end, 2], label="trend", linewidth=2)
    lines!(ax1, pce_data[2:end], label="PCE data", color=(:black, 0.6), linestyle=:dot)
    lines!(ax2, hist_ts[2:end, 1], label="volatility", linewidth=2)    

    fig
end

## MCMC ALGORITHM #############################################################

using Turing
using AdvancedMH

extra0 = nothing
extras = [nothing for _ in 1:length(observations)]

rng = MersenneTwister(1234)
rbpf = RBPF(KalmanFilter(), 256, 0.75)

@model function UCMA(data)
    # set the priors according to the paper
    ψ ~ truncated(Normal(0.0, 1.0); lower=-0.99, upper=0.99)
    γ ~ InverseGamma(10, 0.45)
    σ ~ InverseGamma(10, 0.18)

    # define the UC-MA model
    ucma_model = UC(MA([ψ]), σ, γ)

    # filter the latent states grab the log-likelihood
    (_, _, _), ll = AnalyticalFilters.filter(
        rng, ucma_model, rbpf, data, extra0, extras
    )

    Turing.@addlogprob! ll
end

chains = sample(
    rng,
    UCMA(observations),
    MH(),
    10_000
)

histograms = begin
    fig = Figure(size=(1200, 400))
    for (i, θ) in enumerate([:ψ, :γ, :σ])
        # burn the first 1,000 draws
        CairoMakie.hist(
            fig[1, i],
            chains[θ].data[:,1],
            color = :gray,
            strokewidth = 1,
            bins = 40,
            normalization = :pdf,
            label = String(θ)
        )
    end
    fig
end
