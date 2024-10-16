using BenchmarkTools
using CUDA
using LinearAlgebra
using Plots
using ProgressMeter

BATCH_SIZES = 10 .^ (1:7)
D = 2

####################
#### GENERATORS ####
####################

function random_psd(d)
    A = rand(d, d)
    return A * A'
end

function random_psd_batch(d, bs)
    return [random_psd(d) for _ in 1:bs]
end

function random_psd_strided_batch(d, bs)
    As = Array{Float64}(undef, d, d, bs)
    for i in 1:bs
        As[:, :, i] = random_psd(d)
    end
    return As
end

function random_psd_cuda(d)
    A = random_psd(d)
    return cu(A)
end

function random_psd_cuda_batch(d, bs)
    return [random_psd_cuda(d) for _ in 1:bs]
end

function random_psd_cuda_strided_batch(d, bs)
    As = CUDA.rand(d, d, bs)
    return As = CUDA.CUBLAS.gemm_strided_batched('N', 'T', 1.0f0, As, As)
end

#################
#### METHODS ####
#################

function cpu_native(As::Vector{Matrix{Float64}})
    return [cholesky(A) for A in As]
end

function cpu_2d_analytical(As::Array{Float64,3})
    out = Array{Float64}(undef, size(As))
    out[1, 1, :] .= sqrt.(As[1, 1, :])
    out[2, 1, :] .= As[2, 1, :] ./ out[1, 1, :]
    out[1, 2, :] .= out[2, 1, :]
    out[2, 2, :] .= sqrt.(As[2, 2, :] .- out[2, 1, :] .^ 2)
    return out
end

function gpu_strided_lu(As::CuArray{Float32,3}; pivot::Bool=false)
    # Note: not actually strided — just a helper wrapper
    _, _, res = CUDA.CUBLAS.getrf_strided_batched(As, pivot)
    return res
end
gpu_strided_lu_pivot(As::CuArray{Float32,3}) = gpu_strided_lu(As; pivot=true)

function gpu_chol(As::Vector{CuArray{Float32,2,CUDA.DeviceMemory}})
    CUDA.CUSOLVER.potrfBatched!('L', As)
    return As
end

function gpu_2d_analytical(As::CuArray{Float32,3})
    out = CUDA.zeros(Float32, size(As))
    out[1, 1, :] .= sqrt.(As[1, 1, :])
    out[2, 1, :] .= As[2, 1, :] ./ out[1, 1, :]
    out[1, 2, :] .= out[2, 1, :]
    out[2, 2, :] .= sqrt.(As[2, 2, :] .- out[2, 1, :] .^ 2)
    return out
end

benchmark_methods = Dict{Function,Function}(
    cpu_native => random_psd_batch,
    gpu_strided_lu => random_psd_cuda_strided_batch,
    gpu_strided_lu_pivot => random_psd_cuda_strided_batch,
    gpu_chol => random_psd_cuda_batch,
)

# 2D analytical methods
if D == 2
    push!(benchmark_methods, cpu_2d_analytical => random_psd_strided_batch)
    push!(benchmark_methods, gpu_2d_analytical => random_psd_cuda_strided_batch)
end

##################
#### VALIDATE ####
##################

A_test = random_psd(D)
truth = cholesky(A_test).L
truth_32 = convert.(Float32, truth)

As = repeat(A_test, 1, 1, 1)
res = cpu_2d_analytical(As)
res = tril(res[:, :, 1])
println("cpu_2d_analytical: ", res ≈ truth ? "passed" : "failed")

As = cu(repeat(A_test, 1, 1, 1))
res = gpu_2d_analytical(As)
res = tril(Array(res[:, :, 1]))
println("gpu_2d_analytical: ", res ≈ truth_32 ? "passed" : "failed")

As = cu(repeat(A_test, 1, 1, 1))
res = gpu_strided_lu(As; pivot=true)
U = triu(Array(res[:, :, 1]))
L = tril(Array(res[:, :, 1]))
L[diagind(L)] .= 1.0
A_recon = L * U
println("gpu_lu: ", A_recon ≈ A_test ? "passed" : "failed")

As = [cu(A_test)]
res = gpu_chol(As)
res = tril(Array(res[1]))
println("gpu_chol: ", res ≈ truth_32 ? "passed" : "failed")

###################
#### BENCHMARK ####
###################

function benchmark(d, batch_sizes, methods)
    results = Dict{String,Vector{BenchmarkTools.Trial}}()
    @showprogress for (method, initialiser) in methods
        method_results = Vector{BenchmarkTools.Trial}(undef, length(batch_sizes))
        for (i, bs) in enumerate(batch_sizes)
            As = initialiser(d, bs)
            method_results[i] = @benchmark CUDA.@sync ($method)($As)
        end
        results[String(Symbol(method))] = method_results
    end
    return results
end

# Run benchmark
results = benchmark(D, BATCH_SIZES, benchmark_methods)

###################
#### VISUALISE ####
###################

# Remove LU without pivoting
delete!(results, "gpu_strided_lu")

p = plot(;
    yscale=:log10,
    xaxis=:log10,
    xlabel="Batch size",
    ylabel="Time (s)",
    legend=:topleft,
    size=(800, 600),
    title="Cholesky decomposition benchmark",
)
for (method, trials) in results
    plot!(p, BATCH_SIZES, median.(getfield.(trials, :times)) / 10^9; label=string(method))
end
plot(p)
