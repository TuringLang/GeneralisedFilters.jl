using TestItems
using TestItemRunner

@run_package_tests

@testitem "Kalman filter test" begin
    using AnalyticalFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    rng = StableRNG(1234)
    μ0 = rand(rng, 2)
    Σ0 = rand(rng, 2, 2)
    Σ0 = Σ0 * Σ0'  # make Σ0 positive definite
    A = rand(rng, 2, 2)
    b = rand(rng, 2)
    Q = rand(rng, 2, 2)
    Q = Q * Q'  # make Q positive definite
    H = rand(rng, 2, 2)
    c = rand(rng, 2)
    R = rand(rng, 2, 2)
    R = R * R'  # make R positive definite

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

    observations = [rand(rng, 2)]

    kf = KalmanFilter()

    states, ll = AnalyticalFilters.filter(model, kf, observations, nothing, [nothing])

    # Let Z = [X0, X1, Y1] be the joint state vector
    # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
    P = [
        zeros(2, 6)
        A zeros(2, 4)
        zeros(2, 2) H zeros(2, 2)
    ]
    μ_ϵ = [μ0; b; c]
    Σ_ϵ = [
        Σ0 zeros(2, 4)
        zeros(2, 2) Q zeros(2, 2)
        zeros(2, 4) R
    ]

    # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    # Condition on observations using formula for MVN conditional distribution. See: 
    # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
    y = only(observations)
    I_x = 3:4
    I_y = 5:6
    μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
    Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

    @test only(states).μ ≈ μ_X1
    @test only(states).Σ ≈ Σ_X1

    # Exact marginal distribution to test log-likelihood
    μ_Y1 = μ_Z[I_y]
    Σ_Y1 = Σ_Z[I_y, I_y]
    true_ll = logpdf(MvNormal(μ_Y1, Σ_Y1), y)
    @test ll ≈ true_ll
end

@testitem "Forward algorithm test" begin
    using AnalyticalFilters
    using Distributions
    using StableRNGs
    using SSMProblems

    rng = StableRNG(1234)
    α0 = rand(rng, 3)
    α0 = α0 / sum(α0)
    P = rand(rng, 3, 3)
    P = P ./ sum(P; dims=2)

    struct MixtureModelObservation{T} <: SSMProblems.ObservationProcess{T}
        μs::Vector{T}
    end

    function SSMProblems.logdensity(
        obs::MixtureModelObservation, ::Integer, state::Integer, y, extra
    )
        return logpdf(Normal(obs.μs[state], 1.0), y)
    end

    μs = [0.0, 1.0, 2.0]

    dyn = HomogeneousDiscreteLatentDynamics{Int,Float64}(α0, P)
    obs = MixtureModelObservation(μs)
    model = StateSpaceModel(dyn, obs)

    observations = [rand(rng)]

    states, ll = AnalyticalFilters.filter(
        model, ForwardAlgorithm(), observations, nothing, [nothing]
    )

    # Brute force calculations of each conditional path probability p(x_{1:T} | y_{1:T})
    T = 1
    K = 3
    y = only(observations)
    path_probs = Dict{Tuple{Int,Int},Float64}()
    for x0 in 1:K, x1 in 1:K
        prior_prob = α0[x0] * P[x0, x1]
        likelihood = exp(SSMProblems.logdensity(obs, 1, x1, y, nothing))
        path_probs[(x0, x1)] = prior_prob * likelihood
    end
    marginal = sum(values(path_probs))

    filtered_paths = Base.filter(((k, v),) -> k[end] == 1, path_probs)
    @test states[end][1] ≈ sum(values(filtered_paths)) / marginal
    @test ll ≈ log(marginal)
end

@testitem "Kalman-RBPF test" begin
    using AnalyticalFilters
    using Distributions
    using HypothesisTests
    using LinearAlgebra
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

    # Define inner dynamics
    struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
        μ0::Vector{T}
        Σ0::Matrix{T}
        A::Matrix{T}
        b::Vector{T}
        C::Matrix{T}
        Q::Matrix{T}
    end
    AnalyticalFilters.calc_μ0(dyn::InnerDynamics, extra) = dyn.μ0
    AnalyticalFilters.calc_Σ0(dyn::InnerDynamics, extra) = dyn.Σ0
    AnalyticalFilters.calc_A(dyn::InnerDynamics, ::Integer, extra) = dyn.A
    function AnalyticalFilters.calc_b(dyn::InnerDynamics, ::Integer, extra)
        return dyn.b + dyn.C * extra.prev_outer
    end
    AnalyticalFilters.calc_Q(dyn::InnerDynamics, ::Integer, extra) = dyn.Q

    rng = StableRNG(1234)
    μ0 = rand(rng, 4)
    Σ0s = [rand(rng, 2, 2) for _ in 1:2]
    Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
    Σ0 = [
        Σ0s[1] zeros(2, 2)
        zeros(2, 2) Σ0s[2]
    ]
    A = [
        rand(rng, 2, 2) zeros(2, 2)
        rand(rng, 2, 4)
    ]
    # Make mean-reverting
    A /= 3.0
    A[diagind(A)] .= -0.5
    b = rand(rng, 4)
    Qs = [rand(rng, 2, 2) / 10.0 for _ in 1:2]
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(2, 2)
        zeros(2, 2) Qs[2]
    ]
    H = [zeros(2, 2) rand(rng, 2, 2)]
    c = rand(rng, 2)
    R = rand(rng, 2, 2)
    R = R * R' / 3.0  # make R positive definite

    N_particles = 1000
    T = 20

    observations = [rand(rng, 2) for _ in 1:T]
    extra0 = nothing
    extras = [nothing for _ in 1:T]

    # Kalman filtering

    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    kf_states, kf_ll = AnalyticalFilters.filter(
        full_model, KalmanFilter(), observations, extra0, extras
    )

    # Rao-Blackwellised particle filtering

    outer_dyn = AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:2], Σ0[1:2, 1:2], A[1:2, 1:2], b[1:2], Qs[1]
    )
    inner_dyn = InnerDynamics(
        μ0[3:4], Σ0[3:4, 3:4], A[3:4, 3:4], b[3:4], A[3:4, 1:2], Qs[2]
    )
    obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(H[:, 3:4], c, R)
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = RBPF(KalmanFilter(), N_particles, 0.99)
    (xs, zs, log_ws), ll = AnalyticalFilters.filter(
        rng, hier_model, rbpf, observations, extra0, extras
    )

    # Compare log-likelihoods
    println("Kalman filter log-likelihood:", kf_ll)
    println("RBPF log-likelihood:", ll)

    weights = Weights(softmax(log_ws))

    println("ESS: ", 1 / sum(weights .^ 2))
    println("Weighted mean:", sum(xs .* weights))
    println("Vanilla mean:", sum(xs) / N_particles)
    println("Kalman filter mean:", kf_states[T].μ[1:2])

    # Resample outer states
    # resampled_xs = sample(rng, xs, weights, N_particles)
    # println(mean(first.(resampled_xs)))
    # test = ExactOneSampleKSTest(
    #     first.(resampled_xs), Normal(kf_states[T].μ[1], sqrt(kf_states[T].Σ[1, 1]))
    # )
    # @test pvalue(test) > 0.05

    println("Weighted mean:", sum(getproperty.(zs, :μ) .* weights))
    println("Kalman filter mean:", kf_states[T].μ[3:4])

    # Resample inner states and demarginalise
    # resampled_zs = sample(rng, zs, weights, N_particles)
    # resampled_inner = [rand(rng, Normal(p.μ[1], sqrt(p.Σ[1, 1]))) for p in resampled_zs]
    # test = ExactOneSampleKSTest(
    #     resampled_inner, Normal(kf_states[T].μ[3], sqrt(kf_states[T].Σ[3, 3]))
    # )

    # @test pvalue(test) > 0.05
end

@testitem "GPU Kalman-RBPF test" begin
    using AnalyticalFilters
    using Distributions
    using HypothesisTests
    using LinearAlgebra
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

    using CUDA
    using NNlib

    D_outer = 2
    D_inner = 3
    D_obs = 2

    # Define inner dynamics
    struct InnerDynamics{T} <: LinearGaussianLatentDynamics{T}
        μ0::Vector{T}
        Σ0::Matrix{T}
        A::Matrix{T}
        b::Vector{T}
        C::Matrix{T}
        Q::Matrix{T}
    end
    function AnalyticalFilters.batch_calc_μ0s(dyn::InnerDynamics{T}, extra, N) where {T}
        μ0s = CuArray{Float32}(undef, length(dyn.μ0), N)
        return μ0s[:, :] .= cu(dyn.μ0)
    end
    function AnalyticalFilters.batch_calc_Σ0s(
        dyn::InnerDynamics{T}, extra, N::Integer
    ) where {T}
        Σ0s = CuArray{Float32}(undef, size(dyn.Σ0)..., N)
        return Σ0s[:, :, :] .= cu(dyn.Σ0)
    end
    function AnalyticalFilters.batch_calc_As(
        dyn::InnerDynamics, ::Integer, extra, N::Integer
    )
        As = CuArray{Float32}(undef, size(dyn.A)..., N)
        As[:, :, :] .= cu(dyn.A)
        return As
    end
    function AnalyticalFilters.batch_calc_bs(
        dyn::InnerDynamics, ::Integer, extra, N::Integer
    )
        Cs = CuArray{Float32}(undef, size(dyn.C)..., N)
        Cs[:, :, :] .= cu(dyn.C)
        return NNlib.batched_vec(Cs, extra.prev_outer) .+ cu(dyn.b)
    end
    function AnalyticalFilters.batch_calc_Qs(
        dyn::InnerDynamics, ::Integer, extra, N::Integer
    )
        Q = CuArray{Float32}(undef, size(dyn.Q)..., N)
        return Q[:, :, :] .= cu(dyn.Q)
    end

    rng = StableRNG(1234)
    μ0 = rand(rng, D_outer + D_inner)
    Σ0s = [rand(rng, D_outer, D_outer), rand(rng, D_inner, D_inner)]
    Σ0s = [Σ * Σ' for Σ in Σ0s]  # make Σ0 positive definite
    Σ0 = [
        Σ0s[1] zeros(D_outer, D_inner)
        zeros(D_inner, D_outer) Σ0s[2]
    ]
    A = [
        rand(rng, D_outer, D_outer) zeros(D_outer, D_inner)
        rand(rng, D_inner, D_outer + D_inner)
    ]
    # Make mean-reverting
    A /= 3.0
    A[diagind(A)] .= -0.5
    b = rand(rng, D_outer + D_inner)
    Qs = [rand(rng, D_outer, D_outer) / 10.0, rand(rng, D_inner, D_inner) / 10.0]
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(D_outer, D_inner)
        zeros(D_inner, D_outer) Qs[2]
    ]
    H = [zeros(D_obs, D_outer) rand(rng, D_obs, D_inner)]
    c = rand(rng, D_obs)
    R = rand(rng, D_obs, D_obs)
    R = R * R' / 3.0  # make R positive definite

    N_particles = 1000
    T = 10

    observations = [rand(rng, D_obs) for _ in 1:T]
    extra0 = nothing
    extras = [nothing for _ in 1:T]

    # Kalman filtering

    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    kf_states, kf_ll = AnalyticalFilters.filter(
        full_model, KalmanFilter(), observations, extra0, extras
    )

    # Rao-Blackwellised particle filtering

    outer_dyn = AnalyticalFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:D_outer],
        Σ0[1:D_outer, 1:D_outer],
        A[1:D_outer, 1:D_outer],
        b[1:D_outer],
        Qs[1],
    )
    inner_dyn = InnerDynamics(
        μ0[(D_outer + 1):end],
        Σ0[(D_outer + 1):end, (D_outer + 1):end],
        A[(D_outer + 1):end, (D_outer + 1):end],
        b[(D_outer + 1):end],
        A[(D_outer + 1):end, 1:D_outer],
        Qs[2],
    )
    obs = AnalyticalFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = BatchRBPF(BatchKalmanFilter(N_particles), N_particles, 0.8)
    (xs, zs, log_ws), ll = AnalyticalFilters.filter(
        hier_model, rbpf, observations, extra0, extras
    )

    weights = softmax(log_ws)
    reshaped_weights = reshape(weights, (1, length(weights)))

    println("Weighted mean: ", sum(xs[1:D_outer, :] .* reshaped_weights; dims=2))
    println("Kalman filter mean:", kf_states[T].μ[1:D_outer])

    println("Weighted mean: ", sum(zs.μs .* reshaped_weights; dims=2))
    println("Kalman filter mean:", kf_states[T].μ[(D_outer + 1):end])

    println("Kalman log-likelihood: ", kf_ll)
    println("RBPF log-likelihood: ", ll)
end
