module AnalyticalFilters

using AbstractMCMC: AbstractMCMC
using CUDA
using NNlib
import Distributions: MvNormal
import Random: AbstractRNG, default_rng
using SSMProblems

abstract type FilteringAlgorithm end

# Model types
include("models/linear_gaussian.jl")
include("models/discrete.jl")
include("models/hierarchical.jl")

# Filtering/smoothing algorithms
include("algorithms/kalman.jl")
include("algorithms/forward.jl")
include("algorithms/rbpf.jl")

end
