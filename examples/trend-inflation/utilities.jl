using FredData
using CairoMakie
using Dates

## FRED QUERY ##############################################################################

# there are some redundancies here, but I wanted it to clearly reflect the data in the paper
fred = get_data(
    Fred(),
    "PCEPI";
    observation_start="1959-06-01",
    observation_end="2016-01-01",
    units="pca",
    frequency="q",
    aggregation_method="eop",
);

## CALLBACKS ###############################################################################

struct MeanCallback
    μ::Vector
end

# clearly this isn't particularly robust, but is purely for visualization sake
function (c::MeanCallback)(model, filter, step, states, data; kwargs...)
    μz = mean(
        getproperty.(getproperty.(states.filtered.particles, :z), :μ),
        Weights(weights(states.filtered)),
    )

    μx = mean(
        getproperty.(states.filtered.particles, :x), Weights(weights(states.filtered))
    )

    push!(c.μ, [μx; μz])
    return nothing
end

## PLOTTING UTILITIES ######################################################################

# this is essential for plotting dates
date_format(dates) = x -> [Dates.format(dates[floor(Int, i) + 1], "yyyy") for i in x]
