using AnalyticalFilters
using Distributions
using HiddenMarkovModels
using LinearAlgebra

import AnalyticalFilters: ForwardAlgorithm
import HiddenMarkovModels: initialize_forward, obs_logdensities!

struct HHMWrapper{T<:AbstractHMM}
    hmm::T
end

# NOTES:
# - HHMs.jl does not split the the forward function into initialise, predict, and update so
#   the contents of these functions have had to be copied into new definitions
# - HHMs.jl does not have a separate initialisation state so we do not include extra0 in the
#   filtering signature

function filter(model::HHMWrapper, filter::ForwardAlgorithm, obs::Vector, extras)
    # For this model, state corresponds to storage in HHMs.jl and stores the entire history
    # of states, updated in-place
    aug_extra = isnothing(extras[1]) ? (; T=length(obs)) : (; T=length(obs), extras[1]...)
    state, ll = initialise(model, filter, obs[1], aug_extra)
    for i in 2:length(obs)
        y = obs[i]
        state, step_ll = step(model, filter, i, state, y, extras[i])
        ll += step_ll
    end
    return state, ll
end

## Contents copied from HHMs.jl
function initialise(model::HHMWrapper, ::ForwardAlgorithm, obs, extra)
    t1 = 1
    extra..., T = extra
    if length(extra) == 0
        extra = nothing
    end
    hmm = model.hmm

    # Parameters are only used for their lengths and types so fill with dummy vectors
    storage = initialize_forward(hmm, fill(obs, T), fill(extra, T); seq_ends=[T])

    (; α, B, c) = storage

    # Initialization
    Bₜ₁ = view(B, :, t1)
    obs_logdensities!(Bₜ₁, hmm, obs, extra)
    logm = maximum(Bₜ₁)
    Bₜ₁ .= exp.(Bₜ₁ .- logm)

    init = initialization(hmm)
    αₜ₁ = view(α, :, t1)
    αₜ₁ .= init .* Bₜ₁
    c[t1] = inv(sum(αₜ₁))
    lmul!(c[t1], αₜ₁)

    # For some reason, HHMs.jl never actually updates the logL in storage
    logL = -log(c[t1]) + logm

    return storage, logL
end

## Contents copied from HHMs.jl
function step(model::HHMWrapper, ::ForwardAlgorithm, t, storage, obs, extra)
    hmm = model.hmm
    t = t - 1  # HMMs.jl use a different indexing convention for filtering loop

    (; α, B, c) = storage

    Bₜ₊₁ = view(B, :, t + 1)
    # Probably have an off-by-one error compared to HHMs.jl
    obs_logdensities!(Bₜ₊₁, hmm, obs, extra)
    logm = maximum(Bₜ₊₁)
    Bₜ₊₁ .= exp.(Bₜ₊₁ .- logm)

    trans = transition_matrix(hmm, extra)
    αₜ, αₜ₊₁ = view(α, :, t), view(α, :, t + 1)
    mul!(αₜ₊₁, transpose(trans), αₜ)
    αₜ₊₁ .*= Bₜ₊₁
    c[t + 1] = inv(sum(αₜ₊₁))
    lmul!(c[t + 1], αₜ₊₁)

    # Modified to return incrementing rather than updating
    logL = -log(c[t + 1]) + logm

    return storage, logL
end

D = 3

init = rand(D)
init /= sum(init)

trans = rand(D, D)
trans ./= sum(trans; dims=2)

dists = [Normal(randn(), 1.0) for _ in 1:D]

hmm = HMM(init, trans, dists)

T = 10

ys = rand(T)
extras = [nothing for _ in 1:T]

state, ll = filter(HHMWrapper(hmm), ForwardAlgorithm(), ys, extras)
