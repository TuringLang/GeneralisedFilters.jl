using Distributions
using Random
using SSMProblems
using Test
using CUDA

@testset "Forward Simulation" begin
    @testset "Forward simulation without control" begin
        struct UncontrolledLatentDynamics{T} <: LatentDynamics{T,T}
            μ::T
            σ::T
        end
        SSMProblems.distribution(::UncontrolledLatentDynamics; kwargs...) = Normal(0, 1)
        function SSMProblems.distribution(
            dyn::UncontrolledLatentDynamics, ::Integer, prev_state; kwargs...
        )
            return Normal(prev_state + dyn.μ, dyn.σ)
        end

        struct UncontrolledObservationProcess{T} <: ObservationProcess{T,T}
            σ::T
        end
        function SSMProblems.distribution(
            obs::UncontrolledObservationProcess, ::Integer, state; kwargs...
        )
            return Normal(state, obs.σ)
        end

        model = StateSpaceModel(
            UncontrolledLatentDynamics(0.1, 0.2), UncontrolledObservationProcess(0.3)
        )

        rng = MersenneTwister(1234)
        T = 3

        # Sampling with/without rng
        @test sample(rng, model, T) isa Tuple
        @test sample(model, T) isa Tuple

        # GPU tests
        input_array = CUDA.fill(1.0f0, 10)
        output_array = sample(model, T)
        @test eltype(output_array) == eltype(input_array)
    end

    @testset "Forward simulation with control" begin
        struct ControlledLatentDynamics{T} <: LatentDynamics{T,T}
            μ::T
            σ::T
        end
        function SSMProblems.distribution(::ControlledLatentDynamics; σ_init, kwargs...)
            return Normal(0, σ_init)
        end
        function SSMProblems.distribution(
            dyn::ControlledLatentDynamics, step::Integer, prev_state; dts, kwargs...
        )
            dt = dts[step]
            return Normal(prev_state + dyn.μ * dt, dyn.σ * sqrt(dt))
        end

        struct ControlledObservationProcess{T} <: ObservationProcess{T,T}
            σ::T
        end
        function SSMProblems.distribution(
            obs::ControlledObservationProcess, ::Integer, state; kwargs...
        )
            return Normal(state, obs.σ)
        end

        model = StateSpaceModel(
            ControlledLatentDynamics(0.1, 0.2), ControlledObservationProcess(0.3)
        )

        rng = MersenneTwister(1234)
        T = 3

        σ_init = 0.5
        dts = [0.1, 0.2, 0.3]

        # Sampling with/without rng
        @test sample(rng, model, T; σ_init=σ_init, dts=dts) isa Tuple
        @test sample(model, T; σ_init=σ_init, dts=dts) isa Tuple

        # GPU tests
        input_array = CUDA.fill(1.0f0, 10)
        output_array = sample(model, T; σ_init=σ_init, dts=dts)
        @test eltype(output_array) == eltype(input_array)
    end
end
