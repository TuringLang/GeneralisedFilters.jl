using GeneralisedFilters
using SSMProblems
using Distributions

using Random
using StatsBase
using LinearAlgebra

include("utilities.jl")

## LATENT DYNAMICS #########################################################################

function stochastic_volatility(γ::T; dims::Integer=2) where {T<:Real}
    return GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
        zeros(T, dims),
        Matrix{T}(100I, dims, dims),
        Matrix{T}(I, dims, dims),
        zeros(T, dims),
        Matrix(γ * I, dims, dims),
    )
end

struct LocalLevelTrend{T} <: LinearGaussianLatentDynamics{T} end

# everything remains constant except volatility
GeneralisedFilters.calc_μ0(::LocalLevelTrend{T}; kwargs...) where {T} = T[0;]
GeneralisedFilters.calc_Σ0(::LocalLevelTrend{T}; kwargs...) where {T} = T[100;;]
GeneralisedFilters.calc_A(::LocalLevelTrend{T}, ::Integer; kwargs...) where {T} = T[1;;]
GeneralisedFilters.calc_b(::LocalLevelTrend{T}, ::Integer; kwargs...) where {T} = T[0;]

function GeneralisedFilters.calc_Q(
    ::LocalLevelTrend{T}, ::Integer; new_outer, kwargs...
) where {T<:Real}
    return T[exp(new_outer[1]);;]
end

struct SimpleObservation{T} <: LinearGaussianObservationProcess{T} end

GeneralisedFilters.calc_H(::SimpleObservation{T}, ::Integer; kwargs...) where {T} = T[1;;]
GeneralisedFilters.calc_c(::SimpleObservation{T}, ::Integer; kwargs...) where {T} = T[0;]

function GeneralisedFilters.calc_R(
    ::SimpleObservation{T}, ::Integer; new_outer, kwargs...
) where {T<:Real}
    return T[exp(new_outer[2]);;]
end

## TESTING #################################################################################

# only basline UCSV is included in the marginalized version
function UCSV(γ::T) where {T}
    outer_dyn = stochastic_volatility(γ; dims=2)
    inner_dyn = LocalLevelTrend{T}()
    obs = SimpleObservation{T}()

    return HierarchicalSSM(outer_dyn, inner_dyn, obs)
end

rng = MersenneTwister(1234);
observations = [[pce] for pce in fred.data.value];

ucsv_model = UCSV(0.2)
rbpf = RBPF(KalmanFilter(), 1024; threshold=1.0)

mc = MeanCallback([]);
_, ll = GeneralisedFilters.filter(rng, ucsv_model, rbpf, observations; callback=mc);

# ideally I would plot the ancestor smoothed version... cest la vie
ucsv_plots = begin
    fig = Figure(; size=(1200, 500), fontsize=16)

    hist_ts = vcat(mc.μ'...)
    dateticks = date_format(fred.data.date)

    ax = Axis(
        fig[1:2, 1];
        limits=(nothing, (-14, 18)),
        title="Trend Inflation",
        xtickformat=dateticks,
    )
    lines!(ax, vcat(observations...); color=:red, linestyle=:dash)
    lines!(ax, hist_ts[:, 3]; color=:black)

    ax1 = Axis(fig[1, 2]; title="Volatility", xtickformat=dateticks)
    ax2 = Axis(fig[2, 2]; xtickformat=dateticks)
    lines!(ax1, exp.(0.5 * hist_ts[:, 1]); color=:black, label="permanent")
    lines!(ax2, exp.(0.5 * hist_ts[:, 2]); color=:black, label="transitory")

    axislegend(ax1; position=:rt)
    axislegend(ax2; position=:lt)

    fig
end
