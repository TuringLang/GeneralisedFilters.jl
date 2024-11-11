using CUDA
using LinearAlgebra

function batch_qr_kernel(
    Q::CuDeviceArray{T}, R::CuDeviceArray{T}, A::CuDeviceArray{T}, D
) where {T}
    # Get initial index and stride
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # Process matrices in strides
    for batch_idx in index:stride:size(A, 3)
        # Handle first column
        @inbounds begin
            # Copy first column to Q and compute its norm
            norm_0 = T(0.0)
            for i in 1:D
                Q[i, 1, batch_idx] = A[i, 1, batch_idx]
                norm_0 += A[i, 1, batch_idx] * A[i, 1, batch_idx]
            end
            norm_0 = CUDA.sqrt(norm_0)

            # Normalize first column of Q
            if norm_0 > eps(T)
                inv_norm = T(1.0) / norm_0
                for i in 1:D
                    Q[i, 1, batch_idx] *= inv_norm
                end
            end

            # Process remaining columns
            for i in 2:D
                # Copy current A column to Q
                for k in 1:D
                    Q[k, i, batch_idx] = A[k, i, batch_idx]
                end

                # Subtract projections onto previous vectors
                for j in 1:(i - 1)
                    # Compute projection coefficient
                    proj_coeff = T(0.0)
                    for k in 1:D
                        proj_coeff += A[k, i, batch_idx] * Q[k, j, batch_idx]
                    end

                    # Subtract projection
                    for k in 1:D
                        Q[k, i, batch_idx] -= proj_coeff * Q[k, j, batch_idx]
                    end
                end

                # Normalize current column
                norm_i = T(0.0)
                for k in 1:D
                    norm_i += Q[k, i, batch_idx] * Q[k, i, batch_idx]
                end
                norm_i = CUDA.sqrt(norm_i)

                if norm_i > eps(T)
                    inv_norm = T(1.0) / norm_i
                    for k in 1:D
                        Q[k, i, batch_idx] *= inv_norm
                    end
                end
            end

            # Compute R = Q'A
            for i in 1:D
                for j in i:D  # R is upper triangular
                    dot_prod = T(0.0)
                    for k in 1:D
                        dot_prod += A[k, j, batch_idx] * Q[k, i, batch_idx]
                    end
                    R[i, j, batch_idx] = dot_prod
                end
            end
        end
    end
    return nothing
end

function batch_qr(A::CuArray{T,3}; threads=nothing, blocks=nothing) where {T}
    D = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Input matrices must be square"
    N = size(A, 3)

    # Allocate output arrays
    Q = CUDA.zeros(T, D, D, N)
    R = CUDA.zeros(T, D, D, N)

    # Configure kernel launch
    threads_per_block = isnothing(threads) ? 256 : threads
    max_blocks = 256
    num_blocks = isnothing(blocks) ? min(max_blocks, cld(N, threads_per_block)) : blocks

    # Launch kernel
    @cuda threads = threads_per_block blocks = num_blocks batch_qr_kernel(Q, R, A, D)

    return Q, R
end

using CUDA

function batch_lq_kernel(
    L::CuDeviceArray{T}, Q::CuDeviceArray{T}, A::CuDeviceArray{T}, D
) where {T}
    # Get initial index and stride
    index = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # Process matrices in strides
    for batch_idx in index:stride:size(A, 3)
        # Handle first row
        @inbounds begin
            # Copy first row to Q and compute its norm
            norm_0 = T(0.0)
            for j in 1:D
                Q[1, j, batch_idx] = A[1, j, batch_idx]
                norm_0 += A[1, j, batch_idx] * A[1, j, batch_idx]
            end
            norm_0 = CUDA.sqrt(norm_0)

            # Normalize first row of Q
            if norm_0 > eps(T)
                inv_norm = T(1.0) / norm_0
                for j in 1:D
                    Q[1, j, batch_idx] *= inv_norm
                end
            end

            # Process remaining rows
            for i in 2:D
                # Copy current A row to Q
                for j in 1:D
                    Q[i, j, batch_idx] = A[i, j, batch_idx]
                end

                # Subtract projections onto previous vectors
                for k in 1:(i - 1)
                    # Compute projection coefficient
                    proj_coeff = T(0.0)
                    for j in 1:D
                        proj_coeff += A[i, j, batch_idx] * Q[k, j, batch_idx]
                    end

                    # Subtract projection
                    for j in 1:D
                        Q[i, j, batch_idx] -= proj_coeff * Q[k, j, batch_idx]
                    end
                end

                # Normalize current row
                norm_i = T(0.0)
                for j in 1:D
                    norm_i += Q[i, j, batch_idx] * Q[i, j, batch_idx]
                end
                norm_i = CUDA.sqrt(norm_i)

                if norm_i > eps(T)
                    inv_norm = T(1.0) / norm_i
                    for j in 1:D
                        Q[i, j, batch_idx] *= inv_norm
                    end
                end
            end

            # Compute L = AQ'
            for i in 1:D
                for j in 1:i  # L is lower triangular
                    dot_prod = T(0.0)
                    for k in 1:D
                        dot_prod += A[i, k, batch_idx] * Q[j, k, batch_idx]
                    end
                    L[i, j, batch_idx] = dot_prod
                end
            end
        end
    end
    return nothing
end

function batch_lq(A::CuArray{T,3}; threads=nothing, blocks=nothing) where {T}
    D = size(A, 1)
    @assert size(A, 1) == size(A, 2) "Input matrices must be square"
    N = size(A, 3)

    # Allocate output arrays
    Q = CUDA.zeros(T, D, D, N)
    L = CUDA.zeros(T, D, D, N)

    # Configure kernel launch
    threads_per_block = isnothing(threads) ? 256 : threads
    max_blocks = 256
    num_blocks = isnothing(blocks) ? min(max_blocks, cld(N, threads_per_block)) : blocks

    # Launch kernel
    @cuda threads = threads_per_block blocks = num_blocks batch_lq_kernel(L, Q, A, D)

    return L, Q
end

# Test QR decomposition
# A = CuArray{Float32}(rand(Float32, 3, 3, 1))
# Q, R = batch_qr(A)
# qr_true = qr(Array(A[:, :, 1]))
# true_diag_sign = sign.(diag(qr_true.R))
# println("Q Error: ", Array(Q[:, :, 1]) - true_diag_sign .* qr_true.Q)
# println("R Error: ", Array(R[:, :, 1]) - true_diag_sign .* qr_true.R)

# # Test LQ decomposition
# L, Q = batch_lq(A)
# lq_true = lq(Array(A[:, :, 1]))
# true_diag_sign = sign.(diag(lq_true.L))
# println("L Error: ", Array(L[:, :, 1]) - true_diag_sign .* lq_true.L)
# println("Q Error: ", Array(Q[:, :, 1]) - true_diag_sign .* lq_true.Q)

# # Benchmark
# using BenchmarkTools
# @benchmark CUDA.@sync batch_qr($CUDA.rand(Float32, 4, 4, 10^6))
# @benchmark CUDA.@sync batch_lq($CUDA.rand(Float32, 4, 4, 10^6))

# # Profile
# CUDA.@profile batch_qr(CUDA.rand(Float32, 4, 4, 10^6))
