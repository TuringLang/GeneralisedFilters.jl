import SSMProblems: LatentDynamics, ObservationProcess, simulate
export HierarchicalSSM

struct HierarchicalSSM{T<:Real,OD<:LatentDynamics,M<:AbstractStateSpaceModel} <:
       AbstractStateSpaceModel
    outer_dyn::OD
    inner_model::M
end

function HierarchicalSSM(
    outer_dyn::LatentDynamics{LDT}, inner_dyn::LatentDynamics, obs::ObservationProcess
) where {LDT}
    inner_model = StateSpaceModel(inner_dyn, obs)
    T = promote_type(eltype(inner_model), eltype(LDT))
    return HierarchicalSSM{T,typeof(outer_dyn),typeof(inner_model)}(outer_dyn, inner_model)
end

Base.eltype(::Type{<:HierarchicalSSM{T,ODT,MT}}) where {T,ODT,MT} = T

function AbstractMCMC.sample(
    rng::AbstractRNG, model::HierarchicalSSM, T::Integer; kwargs...
)
    outer_dyn, inner_model = model.outer_dyn, model.inner_model

    zs = Vector{eltype(inner_model.dyn)}(undef, T)
    xs = Vector{eltype(outer_dyn)}(undef, T)
    ys = Vector{eltype(inner_model.obs)}(undef, T)

    # Simulate outer dynamics
    x0 = simulate(rng, outer_dyn; kwargs...)
    z0 = simulate(rng, inner_model.dyn; new_outer=x0, kwargs...)
    for t in 1:T
        prev_x = t == 1 ? x0 : xs[t - 1]
        xs[t] = simulate(rng, model.outer_dyn, t, prev_x; kwargs...)
        zs[t] = simulate(
            rng, inner_model.dyn, t, z0; prev_outer=prev_x, new_outer=xs[t], kwargs...
        )
        ys[t] = simulate(rng, inner_model.obs, t, zs[t]; new_outer=xs[t], kwargs...)
    end

    return x0, z0, xs, zs, ys
end

# TODO: move to SSMProblems
function batch_simulate end
