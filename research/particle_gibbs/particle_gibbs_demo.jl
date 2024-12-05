using GeneralisedFilters
using Distributions
using LinearAlgebra
using LogExpFunctions
using ProgressMeter
using OffsetArrays
using Random
using SSMProblems
using StatsBase

function rand_cov(rng::AbstractRNG, T::Type{<:Real}, d::Int)
    Σ = rand(rng, T, d, d)
    return Σ * Σ'
end

SEED = 1234
Dx = 1
Dy = 1
K = 3
T = Float64
N_particles = 100
N_burnin = 1
N_sample = 100000

rng = MersenneTwister(SEED)

b_prior = MvNormal(zeros(T, Dx), 0.01I)
# b_true = rand(rng, b_prior)
b_true = [-2.0]

# Generate model matrices/vectors
μ0 = rand(rng, T, Dx)
Σ0 = rand_cov(rng, T, Dx)
A = rand(rng, T, Dx, Dx)
Q = rand_cov(rng, T, Dx)
H = rand(rng, T, Dy, Dx)
c = rand(rng, T, Dy)
R = rand_cov(rng, T, Dy)

# Define full model and sample observations
full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_true, Q, H, c, R)
_, _, ys = sample(rng, full_model, K)

# Brute force compute posterior mean using grid
b_grid = range(-5.0, 5.0; length=10000)
log_posteriors = zeros(length(b_grid))
@showprogress for (i, b) in enumerate(b_grid)
    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, [b], Q, H, c, R)
    state, ll = GeneralisedFilters.filter(rng, model, KalmanFilter(), ys)
    log_posteriors[i] = ll + logpdf(b_prior, [b])
end
posteriors = softmax(log_posteriors)
posterior_mean = sum(b_grid .* posteriors)
println("Brute force posterior mean: ", posterior_mean)

# Define augemented dynamics
μ0_aug = [μ0; b_prior.μ]
Σ0_aug = [
    Σ0 zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) b_prior.Σ
]
A_aug = [
    A I
    zeros(T, Dx, Dx) I
]
b_aug = zeros(T, 2 * Dx)
Q_aug = [
    Q zeros(T, Dx, Dx)
    zeros(T, Dx, Dx) zeros(T, Dx, Dx)
]
H_aug = [H zeros(T, Dy, Dx)]

# Calculate ground truth distribution
aug_model = create_homogeneous_linear_gaussian_model(
    μ0_aug, Σ0_aug, A_aug, b_aug, Q_aug, H_aug, c, R
)
state, _ = GeneralisedFilters.filter(rng, aug_model, KalmanFilter(), ys)

println("Ground truth: ", b_true)
println("Ground truth posterior mean: ", state.μ[(Dx + 1):end])

##########################################
#### Particle Metropolis-within-Gibbs ####
##########################################

function log_density(b, ref_traj, A, Q)
    log_prior = logpdf(b_prior, b)
    # log_prior = 0.0
    log_likelihood = 0.0
    for t in 0:(K - 1)
        log_likelihood += logpdf(MvNormal(A * ref_traj[t] + b, Q), ref_traj[t + 1])
    end
    return log_prior + log_likelihood
end

N_steps = N_burnin + N_sample
bf = BF(N_particles; threshold=1.0)
ref_traj = nothing

b_samples = Vector{Vector{T}}(undef, N_sample)
# b_curr = rand(rng, b_prior)
b_curr = [posterior_mean]

ϵ = T(0.1)

@showprogress for i in 1:N_steps
    ### x | θ, y (CSMC)

    # Create model
    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b_curr, Q, H, c, R)

    cb = GeneralisedFilters.DenseAncestorCallback(Vector{T})
    bf_state, _ = GeneralisedFilters.filter(model, bf, ys; ref_state=ref_traj, callback=cb)
    weights = softmax(bf_state.log_weights)
    sampled_idx = sample(1:length(weights), Weights(weights))
    global ref_traj = GeneralisedFilters.get_ancestry(cb.container, sampled_idx)

    ### θ | x, y (MH)
    b_prop = b_curr .+ ϵ * randn(rng, T, Dx)
    log_ratio = log_density(b_prop, ref_traj, A, Q) - log_density(b_curr, ref_traj, A, Q)
    if log(rand(rng)) < log_ratio
        global b_curr = b_prop
    end
    if i > N_burnin
        b_samples[i - N_burnin] = deepcopy(b_curr)
    end

    # Replace with closed form update using Gaussianity
    # Prior is N(0, 0.01)
    # Likelihood is product of N(b; A * x{t-1} + b - x{t}, Q)
    # μ_prior = 0.0
    # σ2_prior = 0.01
    # σ2 = only(Q)
    # xs = [only(A * ref_traj[t - 1] - ref_traj[t]) for t in 1:K]
    # n = K

    # μ_post = (sum(xs) / σ2 + μ_prior / σ2_prior) / (n / σ2 + 1 / σ2_prior)
    # σ2_post = 1 / (n / σ2 + 1 / σ2_prior)

    # global b_curr = [rand(rng, Normal(μ_post, sqrt(σ2_post)))]
    # if i > N_burnin
    #     b_samples[i - N_burnin] = deepcopy(b_curr)
    # end
end

println("Posterior mean: ", mean(b_samples))
println("Posterior mean trimmed: ", mean(b_samples[1000:end]))

# Plot chain
using Plots

plot(only.(b_samples); label="Chain", xlabel="Iteration", ylabel="b_outer", legend=:topleft)
