function create_linear_gaussian_model(
    rng::AbstractRNG, Dx::Integer, Dy::Integer, T::Type{<:Real}=Float64
)
    μ0 = rand(rng, T, Dx)
    Σ0 = rand_cov(rng, T, Dx)
    A = rand(rng, T, Dx, Dx)
    b = rand(rng, T, Dx)
    Q = rand_cov(rng, T, Dx)
    H = rand(rng, T, Dy, Dx)
    c = rand(rng, T, Dy)
    R = rand_cov(rng, T, Dy)

    return create_homogeneous_linear_gaussian_model(μ0, Σ0, A, b, Q, H, c, R)
end

function _compute_joint(model, T::Integer)
    (; μ0, Σ0, A, b, Q) = model.dyn
    (; H, c, R) = model.obs
    Dy, Dx = size(H)

    # Let Z = [X0, X1, ..., XT, Y1, ..., YT] be the joint state vector
    # Write Z = P.Z + ϵ, where ϵ ~ N(μ_ϵ, Σ_ϵ)
    P = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    for t in 1:T
        iA = t * Dx + 1
        jA = (t - 1) * Dx + 1
        P[iA:(iA + Dx - 1), jA:(jA + Dx - 1)] = A

        iH = Dx * (T + 1) + (t - 1) * Dy + 1
        jH = Dx * t + 1
        P[iH:(iH + Dy - 1), jH:(jH + Dx - 1)] = H
    end

    μ_ϵ = zeros(Dx + T * (Dx + Dy))
    μ_ϵ[1:Dx] .= μ0
    for t in 1:T
        ib = t * Dx + 1
        μ_ϵ[ib:(ib + Dx - 1)] = b

        ic = Dx * (T + 1) + (t - 1) * Dy + 1
        μ_ϵ[ic:(ic + Dy - 1)] = c
    end

    Σ_ϵ = zeros(Dx + T * (Dx + Dy), Dx + T * (Dx + Dy))
    Σ_ϵ[1:Dx, 1:Dx] .= Σ0
    for t in 1:T
        iQ = t * Dx + 1
        Σ_ϵ[iQ:(iQ + Dx - 1), iQ:(iQ + Dx - 1)] = Q

        iR = Dx * (T + 1) + (t - 1) * Dy + 1
        Σ_ϵ[iR:(iR + Dy - 1), iR:(iR + Dy - 1)] = R
    end

    # Note (I - P)Z = ϵ and solve for Z ~ N(μ_Z, Σ_Z)
    μ_Z = (I - P) \ μ_ϵ
    Σ_Z = ((I - P) \ Σ_ϵ) / (I - P)'

    return μ_Z, Σ_Z
end