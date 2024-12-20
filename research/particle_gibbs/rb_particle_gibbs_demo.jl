using GeneralisedFilters
using CUDA
using Distributions
using LinearAlgebra
using NNlib
using ProgressMeter
using OffsetArrays
using Random
using SSMProblems
using StatsBase

struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
    μ0::Vector{T}
    Σ0::Matrix{T}
    A::Matrix{T}
    b::Vector{T}
    C::Matrix{T}
    Q::Matrix{T}
end

function GeneralisedFilters.batch_calc_μ0s(dyn::InnerDynamics{T}, N; kwargs...) where {T}
    μ0s = CuArray{T}(undef, length(dyn.μ0), N)
    return μ0s[:, :] .= cu(dyn.μ0)
end

function GeneralisedFilters.batch_calc_Σ0s(
    dyn::InnerDynamics{T}, N::Integer; kwargs...
) where {T}
    Σ0s = CuArray{T}(undef, size(dyn.Σ0)..., N)
    return Σ0s[:, :, :] .= cu(dyn.Σ0)
end

function GeneralisedFilters.batch_calc_As(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    As = CuArray{T}(undef, size(dyn.A)..., N)
    As[:, :, :] .= cu(dyn.A)
    return As
end

function GeneralisedFilters.batch_calc_bs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; prev_outer, kwargs...
) where {T}
    Cs = CuArray{T}(undef, size(dyn.C)..., N)
    Cs[:, :, :] .= cu(dyn.C)
    return NNlib.batched_vec(Cs, prev_outer) .+ cu(dyn.b)
end

function GeneralisedFilters.batch_calc_Qs(
    dyn::InnerDynamics{T}, ::Integer, N::Integer; kwargs...
) where {T}
    Q = CuArray{T}(undef, size(dyn.Q)..., N)
    return Q[:, :, :] .= cu(dyn.Q)
end

function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
    Σ = rand(rng, T, d, d)
    return Σ * Σ'
end

SEED = 1234
D_outer = 1
D_inner = 1
Dy = 2
K = 30
T = Float32
N_particles = 1000
N_burnin = 1
N_sample = 100000

rng = MersenneTwister(SEED)

b_outer_prior = MvNormal(zeros(T, D_outer), I)
b_true = [rand(rng, b_outer_prior); zeros(T, D_inner)]

# Generate model matrices/vectors
μ0 = rand(rng, T, D_outer + D_inner)
Σ0s = [
    rand_cov(rng, T, D_outer) zeros(T, D_outer, D_inner)
    zeros(T, D_inner, D_outer) rand_cov(rng, T, D_inner)
]
A = [
    rand(rng, T, D_outer, D_outer) zeros(T, D_outer, D_inner)
    rand(rng, T, D_inner, D_outer) rand(rng, T, D_inner, D_inner)
]
Q =
    [
        rand_cov(rng, T, D_outer) zeros(T, D_outer, D_inner)
        zeros(T, D_inner, D_outer) rand_cov(rng, T, D_inner)
    ] / T(10.0)
H = [zeros(T, Dy, D_outer) rand(rng, T, Dy, D_inner)]
c = rand(rng, T, Dy)
R = rand_cov(rng, T, Dy) / T(10.0)

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0s, A, b_true, Q, H, c, R)
_, _, ys = sample(rng, full_model, K)

# Define augemented dynamics
μ0_aug = [μ0; b_outer_prior.μ]
Σ0_aug = [
    Σ0s zeros(T, D_outer + D_inner, D_outer)
    zeros(T, D_outer, D_outer + D_inner) b_outer_prior.Σ
]
A_aug = [
    A [I; zeros(T, D_inner, D_outer)]
    zeros(T, D_outer, D_outer + D_inner) I
]
b_aug = zeros(T, D_outer + D_inner + D_outer)
Q_aug = [
    Q zeros(T, D_outer + D_inner, D_outer)
    zeros(T, D_outer, D_outer + D_inner) zeros(T, D_outer, D_outer)
]
H_aug = [H zeros(T, Dy, D_outer)]

# Calculate ground truth distribution
aug_model = create_homogeneous_linear_gaussian_model(
    μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c, R
)
state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)

println("Ground truth: ", b_true[1:D_outer])
println("Ground truth posterior mean: ", state.μ[end])
println("Ground truth posterior variance: ", state.Σ[end, end])

##########################################
#### Particle Metropolis-within-Gibbs ####
##########################################

# Setup
particle_template = GeneralisedFilters.RaoBlackwellisedParticle(
    CuArray{Float32}(undef, D_outer, N_particles),
    GeneralisedFilters.BatchGaussianDistribution(
        CuArray{Float32}(undef, D_inner, N_particles),
        CuArray{Float32}(undef, D_inner, D_inner, N_particles),
    ),
)
particle_type = typeof(particle_template)

function log_density(b_outer, ref_traj, A, Q)
    log_prior = logpdf(b_outer_prior, b_outer)
    log_likelihood = 0.0
    for t in 1:(K - 1)
        log_likelihood += logpdf(MvNormal(A * ref_traj[t] + b_outer, Q), ref_traj[t + 1])
    end
    return log_prior + log_likelihood
end

N_steps = N_burnin + N_sample
M = floor(Int64, N_particles * log(N_particles))
rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles; threshold=1.0)
ref_traj = nothing

b_outer_samples = Vector{CuArray{Float32}}(undef, N_sample)
# b_outer_curr = rand(rng, b_outer_prior)
b_outer_curr = [state.μ[end]]

ϵ = T(0.5)
n_accept = 0
acceptance_rate = T(0.0)
target_acceptance_rate = T(0.25)
adaptation_rate = T(0.01)
N_ADAPT = 4000

prog = Progress(N_steps, 1)
for i in 1:N_steps
    ### x | θ, y (CSMC)

    # Create model
    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:D_outer],
        Σ0s[1:D_outer, 1:D_outer],
        A[1:D_outer, 1:D_outer],
        b_outer_curr,  # set current value
        Q[1:D_outer, 1:D_outer],
    )
    inner_dyn = InnerDynamics(
        μ0[(D_outer + 1):end],
        Σ0s[(D_outer + 1):end, (D_outer + 1):end],
        A[(D_outer + 1):end, (D_outer + 1):end],
        zeros(T, D_inner),
        A[(D_outer + 1):end, 1:D_outer],
        Q[(D_outer + 1):end, (D_outer + 1):end],
    )
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    tree = GeneralisedFilters.ParallelParticleTree(deepcopy(particle_template), M)
    cb = GeneralisedFilters.ParallelAncestorCallback(tree)
    rbpf_state, _ = GeneralisedFilters.filter(
        hier_model, rbpf, ys; ref_state=ref_traj, callback=cb
    )
    weights = softmax(rbpf_state.log_weights)
    sampled_idx = CUDA.@allowscalar sample(1:length(weights), Weights(weights))
    global ref_traj = GeneralisedFilters.get_ancestry(tree, sampled_idx, K)

    ref_traj_outer_cpu = map(r -> Array(dropdims(r.x_particles; dims=2)), ref_traj)

    ### θ | x, y (MH)
    b_outer_prop = b_outer_curr .+ ϵ * randn(rng, T, D_outer)
    log_ratio =
        log_density(
            b_outer_prop,
            ref_traj_outer_cpu,
            A[1:D_outer, 1:D_outer],
            Q[1:D_outer, 1:D_outer],
        ) - log_density(
            b_outer_curr,
            ref_traj_outer_cpu,
            A[1:D_outer, 1:D_outer],
            Q[1:D_outer, 1:D_outer],
        )
    if log(rand(rng)) < log_ratio
        global b_outer_curr = b_outer_prop
        n_accept += 1
        acceptance_rate = T(n_accept / i)
    end
    if i > N_burnin
        b_outer_samples[i - N_burnin] = deepcopy(b_outer_curr)
    end

    # Adapt step size
    if i <= N_ADAPT
        ϵ *= exp(adaptation_rate * (acceptance_rate - target_acceptance_rate))
    end

    # Update progress bar with acceptance rate
    ProgressMeter.next!(prog; showvalues=[("Acceptance rate", acceptance_rate), ("ϵ", ϵ)])
end

println("Posterior mean: ", mean(b_outer_samples[1000:end]))

# Plot chain
using Plots

b_outer_samples_flat = map(b -> only(Array(b)), b_outer_samples)

plot(
    b_outer_samples_flat;
    label="Chain",
    xlabel="Iteration",
    ylabel="b_outer",
    legend=:topleft,
)

# Ground truth: Float32[-0.81195045]
# Ground truth posterior mean: -0.8751777
# Ground truth posterior variance: 0.003282427
# Progress: 100% Time: 2:37:58
#   Acceptance rate:  0.22514576
#   ϵ:                0.2409957
# Posterior mean: Float32[-0.87571603]

savefig("research/particle_gibbs/rb_particle_gibbs_demo.png")

# julia> var(b_outer_samples_flat)
# 0.0033998052f0

# julia> state.Σ[end, end]
# 0.003282427f0
