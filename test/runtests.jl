using TestItems
using TestItemRunner

@run_package_tests

include("batch_kalman_test.jl")

@testitem "Kalman filter test" begin
    using GeneralisedFilters
    using Distributions
    using LinearAlgebra
    using StableRNGs

    for Dy in [2, 3, 4]
        Dx = 3

        rng = StableRNG(1234)
        μ0 = rand(rng, Dx)
        Σ0 = rand(rng, Dx, Dx)
        Σ0 = Σ0 * Σ0'  # make Σ0 positive definite
        A = rand(rng, Dx, Dx)
        b = rand(rng, Dx)
        Q = rand(rng, Dx, Dx)
        Q = Q * Q'  # make Q positive definite
        H = rand(rng, Dy, Dx)
        c = rand(rng, Dy)
        R = rand(rng, Dy, Dy)
        R = R * R'  # make R positive definite

        model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)

        observations = [rand(rng, Dy)]

        kf = KalmanFilter()

        states, ll = GeneralisedFilters.filter(rng, model, kf, observations)

        # Let Z = [X0, X1, Y1] be the joint state vector
        # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
        P = [
            zeros(Dx, 2Dx + Dy)
            A zeros(Dx, Dx + Dy)
            zeros(Dy, Dx) H zeros(Dy, Dy)
        ]
        μ_ϵ = [μ0; b; c]
        Σ_ϵ = [
            Σ0 zeros(Dx, Dx + Dy)
            zeros(Dx, Dx) Q zeros(Dx, Dy)
            zeros(Dy, 2Dx) R
        ]

        # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
        μ_Z = (I - P) \ μ_ϵ
        Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

        # Condition on observations using formula for MVN conditional distribution. See: 
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Conditional_distributions
        y = only(observations)
        I_x = (Dx + 1):(2Dx)
        I_y = (2Dx + 1):(2Dx + Dy)
        μ_X1 = μ_Z[I_x] + Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ (y - μ_Z[I_y]))
        Σ_X1 = Σ_Z[I_x, I_x] - Σ_Z[I_x, I_y] * (Σ_Z[I_y, I_y] \ Σ_Z[I_y, I_x])

        @test states.μ ≈ μ_X1
        @test states.Σ ≈ Σ_X1

        # Exact marginal distribution to test log-likelihood
        μ_Y1 = μ_Z[I_y]
        Σ_Y1 = Σ_Z[I_y, I_y]
        LinearAlgebra.hermitianpart!(Σ_Y1)
        true_ll = logpdf(MvNormal(μ_Y1, Σ_Y1), y)
        @test ll ≈ true_ll
    end
end

@testitem "Bootstrap filter test" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp

    T = Float32
    rng = StableRNG(1234)
    σx², σy² = randexp(rng, T, 2)

    # initial state distribution
    μ0 = zeros(T, 2)
    Σ0 = PDMat(T[1 0; 0 1])

    # state transition equation
    A = T[1 1; 0 1]
    b = T[0; 0]
    Q = PDiagMat([σx²; 0])

    # observation equation
    H = T[1 0]
    c = T[0;]
    R = [σy²;;]

    # when working with PDMats, the Kalman filter doesn't play nicely without this
    function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
        return PDMat(Symmetric(mat))
    end

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    _, _, data = sample(rng, model, 20)

    bf = BF(2^12; threshold=0.8)
    apf = APF(2^10; threshold=1.0)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, data)
    _, llapf = GeneralisedFilters.filter(rng, model, apf, data)
    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), data)

    xs = bf_state.filtered.particles
    ws = softmax(bf_state.filtered.log_weights)

    # Compare filtered states
    @test first(kf_state.μ) ≈ sum(first.(xs) .* ws) rtol = 1e-2

    # since this is log valued, we can up the tolerance
    @test llkf ≈ llbf atol = 0.1
    @test llkf ≈ llapf atol = 2
end

@testitem "APF filter test" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using Random: randexp

    T = Float32
    rng = StableRNG(1234)
    σx², σy² = randexp(rng, T, 2)

    # initial state distribution
    μ0 = zeros(T, 2)
    Σ0 = PDMat(T[1 0; 0 1])

    # state transition equation
    A = T[1 1; 0 1]
    b = T[0; 0]
    Q = PDiagMat([σx²; 0])

    # observation equation
    H = T[1 0]
    c = T[0;]
    R = [σy²;;]

    # when working with PDMats, the Kalman filter doesn't play nicely without this
    function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
        return PDMat(Symmetric(mat))
    end

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    _, _, data = sample(rng, model, 20)

    bf = APF(2^10; threshold=0.8)
    _, llbf = GeneralisedFilters.filter(rng, model, bf, data)
    _, llkf = GeneralisedFilters.filter(rng, model, KF(), data)

    # since this is log valued, we can up the tolerance
    @test llkf ≈ llbf atol = 2
end

@testitem "Guided particle filter test" begin
    using GeneralisedFilters
    using SSMProblems
    using StableRNGs
    using PDMats
    using LinearAlgebra
    using LogExpFunctions: softmax
    using Random: randexp
    using Distributions

    # this is a pseudo optimal proposal kernel for linear Gaussian models
    struct LinearGaussianProposal{T<:Real} <: GeneralisedFilters.AbstractProposal
        φ::Vector{T}
    end

    # a lot of computations done at each step
    function SSMProblems.distribution(
        model::AbstractStateSpaceModel,
        kernel::LinearGaussianProposal,
        step::Integer,
        state,
        observation;
        kwargs...,
    )
        # get model dimensions
        dx = length(state)
        dy = length(observation)

        # see (Corenflos et al, 2021) for details
        A = GeneralisedFilters.calc_A(model.dyn, step; kwargs...)
        Γ = diagm(dx, dy, kernel.φ[(dx + 1):end])
        Σ = PDiagMat(φ[1:dx])

        return MvNormal(inv(Σ) * A * state + inv(Σ) * Γ * observation, Σ)
    end

    T = Float32
    rng = StableRNG(1234)
    σx², σy² = randexp(rng, T, 2)

    # initial state distribution
    μ0 = zeros(T, 1)
    Σ0 = PDiagMat(T[1;])

    # state transition equation
    A = T[0.9;;]
    b = T[0;]
    Q = PDiagMat([σx²;])

    # observation equation
    H = T[1;;]
    c = T[0;]
    R = PDiagMat([σy²;])

    # proposal kernel (kind of optimal...)
    φ = ones(T, 2)
    proposal = LinearGaussianProposal(φ)

    # when working with PDMats, the Kalman filter doesn't play nicely without this
    function Base.convert(::Type{PDMat{T,MT}}, mat::MT) where {MT<:AbstractMatrix,T<:Real}
        return PDMat(Symmetric(mat))
    end

    model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    _, _, data = sample(rng, model, 20)

    # bootstrap filter
    bf = BF(2^12; threshold=0.8)
    bf_state, llbf = GeneralisedFilters.filter(rng, model, bf, data)
    bf_xs = bf_state.filtered.particles
    bf_ws = softmax(bf_state.filtered.log_weights)

    # guided particle filter
    pf = GPF(2^12, proposal; threshold=0.8)
    pf_state, llpf = GeneralisedFilters.filter(rng, model, pf, data)
    pf_xs = pf_state.filtered.particles
    pf_ws = softmax(pf_state.filtered.log_weights)

    kf_state, llkf = GeneralisedFilters.filter(rng, model, KF(), data)

    # Compare filtered states
    @test first(kf_state.μ) ≈ sum(first.(bf_xs) .* bf_ws) rtol = 1e-2
    @test first(kf_state.μ) ≈ sum(first.(pf_xs) .* pf_ws) rtol = 1e-2

    # since this is log valued, we can up the tolerance
    @test llkf ≈ llbf atol = 0.1
    @test llkf ≈ llpf atol = 2
end

@testitem "Forward algorithm test" begin
    using GeneralisedFilters
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
        obs::MixtureModelObservation{T},
        step::Integer,
        state::Integer,
        observation;
        kwargs...,
    ) where {T}
        return logpdf(Normal(obs.μs[state], one(T)), observation)
    end

    μs = [0.0, 1.0, 2.0]

    dyn = HomogeneousDiscreteLatentDynamics{Int,Float64}(α0, P)
    obs = MixtureModelObservation(μs)
    model = StateSpaceModel(dyn, obs)

    observations = [rand(rng)]

    fw = ForwardAlgorithm()
    state, ll = GeneralisedFilters.filter(model, fw, observations)

    # Brute force calculations of each conditional path probability p(x_{1:T} | y_{1:T})
    T = 1
    K = 3
    y = only(observations)
    path_probs = Dict{Tuple{Int,Int},Float64}()
    for x0 in 1:K, x1 in 1:K
        prior_prob = α0[x0] * P[x0, x1]
        likelihood = exp(SSMProblems.logdensity(obs, 1, x1, y))
        path_probs[(x0, x1)] = prior_prob * likelihood
    end
    marginal = sum(values(path_probs))

    filtered_paths = Base.filter(((k, v),) -> k[end] == 1, path_probs)
    @test state[1] ≈ sum(values(filtered_paths)) / marginal
    @test ll ≈ log(marginal)
end

@testitem "Kalman-RBPF test" begin
    using GeneralisedFilters
    using Distributions
    using HypothesisTests
    using LinearAlgebra
    using LogExpFunctions: softmax
    using StableRNGs
    using StatsBase

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
    GeneralisedFilters.calc_μ0(dyn::InnerDynamics; kwargs...) = dyn.μ0
    GeneralisedFilters.calc_Σ0(dyn::InnerDynamics; kwargs...) = dyn.Σ0
    GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
    function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
        return dyn.b + dyn.C * prev_outer
    end
    GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

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
    Qs = [rand(rng, D_outer, D_outer), rand(rng, D_inner, D_inner)] ./ 10.0
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
    T = 20

    observations = [rand(rng, D_obs) for _ in 1:T]

    # Kalman filtering

    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    kf_states, kf_ll = GeneralisedFilters.filter(
        rng, full_model, KalmanFilter(), observations
    )

    # Rao-Blackwellised particle filtering

    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
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
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = RBPF(
        KalmanFilter(),
        N_particles;
        threshold=0.8,
        resampler=GeneralisedFilters.Multinomial(),
    )
    states, ll = GeneralisedFilters.filter(rng, hier_model, rbpf, observations)

    # Extract final filtered states
    xs = map(p -> getproperty(p, :x), states.filtered.particles)
    zs = map(p -> getproperty(p, :z), states.filtered.particles)
    log_ws = states.filtered.log_weights

    # Compare log-likelihoods
    # println("KF log-likelihood:\t", kf_ll)
    # println("RBPF log-likelihood:\t", ll)

    @test kf_ll ≈ ll rtol = 1e-2

    weights = Weights(softmax(log_ws))

    # println("ESS: ", 1 / sum(weights .^ 2))
    # println("Weighted mean:", sum(xs .* weights))
    # println("Kalman filter mean:", kf_states.μ[1:2])

    # Higher tolerance for outer state since variance is higher
    @test first(kf_states.μ) ≈ sum(first.(xs) .* weights) rtol = 1e-1

    # Resample outer states
    # resampled_xs = sample(rng, xs, weights, N_particles)
    # println(mean(first.(resampled_xs)))
    # test = ExactOneSampleKSTest(
    #     first.(resampled_xs), Normal(kf_states[T].μ[1], sqrt(kf_states[T].Σ[1, 1]))
    # )
    # @test pvalue(test) > 0.05

    # println("Weighted mean:", sum(getproperty.(zs, :μ) .* weights))
    # println("Kalman filter mean:", kf_states.μ[3:4])

    @test last(kf_states.μ) ≈ sum(last.(getproperty.(zs, :μ)) .* weights) rtol = 1e-2

    # Resample inner states and demarginalise
    # resampled_zs = sample(rng, zs, weights, N_particles)
    # resampled_inner = [rand(rng, Normal(p.μ[1], sqrt(p.Σ[1, 1]))) for p in resampled_zs]
    # test = ExactOneSampleKSTest(
    #     resampled_inner, Normal(kf_states[T].μ[3], sqrt(kf_states[T].Σ[3, 3]))
    # )

    # @test pvalue(test) > 0.05
end

@testitem "GPU Kalman-RBPF test" begin
    using GeneralisedFilters
    using CUDA
    using NNlib
    using LinearAlgebra
    using StableRNGs

    # TODO: seems to pass when D_inner = D_obs but fails otherwise
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
    function GeneralisedFilters.batch_calc_μ0s(
        dyn::InnerDynamics{T}, N; kwargs...
    ) where {T}
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
    Qs = [rand(rng, D_outer, D_outer), rand(rng, D_inner, D_inner)] ./ 10.0
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(D_outer, D_inner)
        zeros(D_inner, D_outer) Qs[2]
    ]
    H = [zeros(D_obs, D_outer) rand(rng, D_obs, D_inner)]
    c = rand(rng, D_obs)
    R = rand(rng, D_obs, D_obs)
    R = R * R' / 3.0  # make R positive definite

    N_particles = 2000
    T = 20

    observations = [rand(rng, D_obs) for _ in 1:T]

    # Kalman filtering

    full_model = create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
    kf_state, kf_ll = GeneralisedFilters.filter(full_model, KalmanFilter(), observations)

    # Rao-Blackwellised particle filtering

    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
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
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(
        H[:, (D_outer + 1):end], c, R
    )
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = BatchRBPF(
        BatchKalmanFilter(N_particles), N_particles; threshold=0.8, resampler=Multinomial()
    )
    states, ll = GeneralisedFilters.filter(hier_model, rbpf, observations)

    # Extract final filtered states
    xs = states.filtered.x_particles
    zs = states.filtered.z_particles
    log_ws = states.filtered.log_weights

    weights = softmax(log_ws)
    reshaped_weights = reshape(weights, (1, length(weights)))

    # println("Weighted mean: ", sum(xs[1:D_outer, :] .* reshaped_weights; dims=2))
    # println("Kalman filter mean:", kf_state.μ[1:D_outer])

    # println("Weighted mean: ", sum(zs.μs .* reshaped_weights; dims=2))
    # println("Kalman filter mean:", kf_state.μ[(D_outer + 1):end])

    # println("Kalman log-likelihood: ", kf_ll)
    # println("RBPF log-likelihood: ", ll)

    @test kf_ll ≈ ll rtol = 1e-2
    @test first(kf_state.μ) ≈ sum(xs[1, :] .* weights) rtol = 1e-1
    @test last(kf_state.μ) ≈ sum(zs.μs[end, :] .* weights) rtol = 1e-2
end

@testitem "RBPF ancestory test" begin
    using GeneralisedFilters
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

    GeneralisedFilters.calc_μ0(dyn::InnerDynamics; kwargs...) = dyn.μ0
    GeneralisedFilters.calc_Σ0(dyn::InnerDynamics; kwargs...) = dyn.Σ0
    GeneralisedFilters.calc_A(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.A
    function GeneralisedFilters.calc_b(dyn::InnerDynamics, ::Integer; prev_outer, kwargs...)
        return dyn.b + dyn.C * prev_outer
    end
    GeneralisedFilters.calc_Q(dyn::InnerDynamics, ::Integer; kwargs...) = dyn.Q

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
    Qs = [rand(rng, 2, 2) for _ in 1:2]
    Qs = [Q * Q' for Q in Qs]  # make Q positive definite
    Q = [
        Qs[1] zeros(2, 2)
        zeros(2, 2) Qs[2]
    ]
    H = [zeros(2, 2) rand(rng, 2, 2)]
    c = rand(rng, 2)
    R = rand(rng, 2, 2)
    R = R * R' / 3.0  # make R positive definite

    N_particles = 100
    T = 20

    observations = [rand(rng, 2) for _ in 1:T]

    # Rao-Blackwellised particle filtering

    outer_dyn = GeneralisedFilters.HomogeneousLinearGaussianLatentDynamics(
        μ0[1:2], Σ0[1:2, 1:2], A[1:2, 1:2], b[1:2], Qs[1]
    )
    inner_dyn = InnerDynamics(
        μ0[3:4], Σ0[3:4, 3:4], A[3:4, 3:4], b[3:4], A[3:4, 1:2], Qs[2]
    )
    obs = GeneralisedFilters.HomogeneousLinearGaussianObservationProcess(H[:, 3:4], c, R)
    hier_model = HierarchicalSSM(outer_dyn, inner_dyn, obs)

    rbpf = RBPF(KalmanFilter(), N_particles; threshold=0.8)
    particle_type = GeneralisedFilters.RaoBlackwellisedContainer{
        eltype(outer_dyn),GeneralisedFilters.rb_eltype(hier_model.inner_model)
    }
    cb = GeneralisedFilters.AncestorCallback(particle_type, N_particles, 1.0)
    states, ll = GeneralisedFilters.filter(rng, hier_model, rbpf, observations; callback=cb)

    tree = cb.tree
    paths = GeneralisedFilters.get_ancestry(tree)

    # TODO: add proper test comparing to dense storage
end
