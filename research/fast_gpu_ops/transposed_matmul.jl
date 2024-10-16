using BenchmarkTools
using CUDA

N = 1000
D = 30

A = CUDA.randn(D, D, N)
B = CUDA.randn(D, D, N)

# Simple kernel to transpose the first two dimensions of a 3D array
function batch_transpose!(B, B_T)
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = blockDim().x * gridDim().x
    for i in index:stride:size(B, 3)
        for j in 1:size(B, 2)
            for k in 1:size(B, 1)
                B_T[k, j, i] = B[j, k, i]
            end
        end
    end
end

# Compute A * B_T by first transposing B explicitly
function transpose_first(A, B)
    B_T = CuArray{Float32}(undef, size(B, 2), size(B, 1), size(B, 3))
    @cuda threads = 256 blocks = ceil(Int, size(B, 3) / 256) batch_transpose!(B, B_T)
    return CUDA.CUBLAS.gemm_strided_batched('N', 'N', A, B_T)
end

# Benchmark the three approaches
bench1 = @benchmark CUDA.@sync CUDA.CUBLAS.gemm_strided_batched('N', 'N', $A, $B)
bench2 = @benchmark CUDA.@sync CUDA.CUBLAS.gemm_strided_batched('N', 'T', $A, $B)
bench3 = @benchmark CUDA.@sync transpose_first($A, $B)

# Print the results
println("Non-transposed gemm: $(median(bench1.times) / 1e3) μs")
println("Transposed gemm    : $(median(bench2.times) / 1e3) μs")
println("Transposed manually: $(median(bench3.times) / 1e3) μs")
