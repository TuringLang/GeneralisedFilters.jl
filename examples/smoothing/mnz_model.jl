using AnalyticalFilters, SSMProblems
using Distributions, Random
using LinearAlgebra, MatrixEquations, Polynomials
using StatsFuns, StatsBase

## MNZ MODEL ##################################################################

# use the UC model from (Morely-Nelson-Zivot, 2003)
function morely_nelson_zivot(
    φ::Vector{T}, ση::T, σε::T, ρ::T, μ::T, init_val::T
) where {T<:Real}
    dyn = AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics(
        [init_val; zeros(T, 2)],
        cat(T[100], lyapd([φ'; 1 0], [σε^2 0; 0 0]), dims=(1,2)),
        [1 0 0; 0 φ'; 0 1 0],
        [μ, 0, 0],
        [ση^2 ρ 0; ρ σε^2 0; 0 0 0]
    )

    obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(
        T[1 1 0],
        zeros(T, 1),
        T[1e-8;;]
    )

    return SSMProblems.StateSpaceModel(dyn, obs)
end

## REPLICATION ################################################################

using Dates, FredData
using CairoMakie

# I usually use `cca`, but it requires some tinkering with FredData.jl
fred_data = get_data(
    Fred(),
    "GDPC1",
    observation_start = "1947-01-01",
    observation_end   = "1998-04-01",
    units = "log",
)

# use PCE instead since quarterly CPI doesn't go back far enough for that
gdp_data  = fred_data.data.value
gdp_dates = fred_data.data.date

observations = [[100*gdp] for gdp in gdp_data]

# just match the parameterization from the paper
mnz_model = morely_nelson_zivot(
    [1.5303, -0.6098], 0.6893, 0.6199, 0.0, 0.8119, 724.0
)

extra0 = nothing
extras = [nothing for _ in 1:length(observations)]

# Kalman filter
filt_states, ll = AnalyticalFilters.filter(
    mnz_model, KalmanFilter(), observations, extra0, extras
)

# Rauch–Tung–Striebel smoother
smooth_states, ll = AnalyticalFilters.smoother(
    mnz_model, KalmanFilter(), observations, extra0, extras
)

# format dates for Makie plots
date_format(dates) = x -> [
    Dates.format(dates[floor(Int, i)+1], "yyyy") for i in x
]

mnz_plots = begin
    xgap_filt   = hcat(getproperty.(filt_states, :μ)...)[2,:]
    xgap_smooth = hcat(getproperty.(smooth_states, :μ)...)[2,:]

    fig = Figure(linewidth=2)

    ax = Axis(
        fig[1, 1],
        title = "Estimated Output Gap",
        xtickformat = date_format(gdp_dates)
    )

    lines!(ax, xgap_filt, label="filtered")
    lines!(ax, xgap_smooth, label="smoothed")
    axislegend()

    fig
end