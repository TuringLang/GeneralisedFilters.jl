export smoother

# TODO: add typing to this function also maybe rename...
function smoother_step(smoothed_state, filtered_state, predicted_state, A)
    μs, Σs = smoothed_state.μ, smoothed_state.Σ
    μf, Σf = filtered_state.μ, filtered_state.Σ
    μpred, Σpred  = predicted_state.μ, predicted_state.Σ

    J = Σf*A'*inv(Σpred)
    μ = μf + J*(μs-μpred)
    Σ = Σf + J*(Σs-Σpred)*J'

    return (;μ, Σ)
end

# not sure if we want to call this smoother
function smoother(
    model::LinearGaussianStateSpaceModel{T},
    algo::KalmanFilter,
    data::Vector{Vector{T}},
    extra0,
    extras,
) where {T}
    states, ll = filter(model, algo, data, extra0, extras)

    N = length(data)
    smoothed_states = Vector{@NamedTuple{μ::Vector{T}, Σ::Matrix{T}}}(undef, N)
    smoothed_states[end] = deepcopy(states[end])

    for step in (N-1):-1:1
        # recompute the predicted states
        A, b, Q = calc_params(model.dyn, step, extras[step])
        predicted_state = (
            μ = A*states[step].μ+b,
            Σ = A*states[step].Σ*A'+Q
        )

        # run the RTS kernel
        smoothed_states[step] = smoother_step(
            smoothed_states[step+1], states[step], predicted_state, A
        )
    end

    return smoothed_states, ll
end