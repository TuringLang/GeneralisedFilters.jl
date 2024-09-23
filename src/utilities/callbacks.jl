import StatsBase: Weights

# RAO-BLACKWELLIZED PARTICLE CLOUD FUNCTIONS ##################################

# function StatsBase.mean(
#         rng::AbstractRNG,
#         model::AbstractStateSpaceModel,
#         algo,
#         state,
#         i::Integer
#     )
#     xs, zs, logws = state
#     return rb_mean(xs, zs, logws)
# end

function rb_mean(xs, zs, logws)
    ws = Weights(softmax(logws))
    return [
        mean(xs, ws);
        sum(getproperty.(zs, :μ), ws) / sum(ws)
    ]
end


# function StatsBase.cov(
#         rng::AbstractRNG,
#         model::AbstractStateSpaceModel,
#         algo,
#         state,
#         i::Integer
#     )
#     xs, zs, logws = state
#     return rb_cov(xs, zs, logws)
# end

function rb_cov(xs, zs, logws)
    return rb_mean_and_cov(xs, zs, logws)[2]
end


# function StatsBase.mean_and_cov(
#         rng::AbstractRNG,
#         model::AbstractStateSpaceModel,
#         algo,
#         state,
#         i::Integer
#     )
#     xs, zs, logws = state
#     return rb_mean_and_cov(xs, zs, logws)
# end

function rb_mean_and_cov(xs, zs, logws)
    ws = Weights(softmax(logws))
    dims = (length(xs[1]) + length(zs[1].μ))

    μ = μd = zeros(dims)
    Σ = zeros(dims, dims)

    for i in eachindex(ws)
        wi = ws[i]
        if wi > 0.0
            # TODO: replace these with Rao-Blackwellized particle routines
            BLAS.axpy!(wi, [xs[i]; getproperty(zs[i], :μ)], μ)
            BLAS.axpy!(wi, cat(0.0, getproperty(zs[i], :Σ), dims=(1,2)), Σ)
        end
    end

    for i in eachindex(ws)
        wi = ws[i]
        if wi > 0.0
            μd = [xs[i]; getproperty(zs[i],:μ)] - μ
            BLAS.axpy!(wi, μd*μd', Σ)
        end
    end

    return μ, Σ
end