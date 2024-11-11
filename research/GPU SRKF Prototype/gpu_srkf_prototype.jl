using CUDA
using LinearAlgebra
using Random

using GeneralisedFilters

include("gpu_qr.jl")

N = 3
Dx = 2
Dy = 2
T = Float32
SEED = 1234

rng = MersenneTwister(SEED)

function rand_psd(rng, T, n)
    A = rand(rng, T, n, n)
    return A'A + T(0.1)I
end

# Generate models
μ0s = [rand(rng, T, Dx) for _ in 1:N]
Σ0s = [rand_psd(rng, T, Dx) for _ in 1:N]
As = [rand(rng, T, Dx, Dx) for _ in 1:N]
bs = [rand(rng, T, Dx) for _ in 1:N]
Qs = [rand_psd(rng, T, Dx) for _ in 1:N]
Hs = [rand(rng, T, Dy, Dx) for _ in 1:N]
cs = [rand(rng, T, Dy) for _ in 1:N]
Rs = [rand_psd(rng, T, Dy) for _ in 1:N]

ys = [rand(rng, T, Dy) for _ in 1:N]

# Compute ground truth smoothing distributions
models = [
    create_homogeneous_linear_gaussian_model(
        μ0s[i], Σ0s[i], As[i], bs[i], Qs[i], Hs[i], cs[i], Rs[i]
    ) for i in 1:N
]
outputs = [
    GeneralisedFilters.filter(rng, model, KalmanFilter(), [ys[i]]) for
    (i, model) in enumerate(models)
]

states, lls = zip(outputs...)

# Convert to batched CuArrays ('... x N' arrays)
μ0s_gpu = CuArray(reshape(hcat(μ0s...), Dx, N))
Σ0s_gpu = CuArray(reshape(hcat(Σ0s...), Dx, Dx, N))
As_gpu = CuArray(reshape(hcat(As...), Dx, Dx, N))
bs_gpu = CuArray(reshape(hcat(bs...), Dx, N))
Qs_gpu = CuArray(reshape(hcat(Qs...), Dx, Dx, N))
Hs_gpu = CuArray(reshape(hcat(Hs...), Dy, Dx, N))
cs_gpu = CuArray(reshape(hcat(cs...), Dy, N))
Rs_gpu = CuArray(reshape(hcat(Rs...), Dy, Dy, N))

# Compute Cholesky factors
# Note: these are meant to be lower, implying A is not upper triangular as would be expected
Σ0s_chol = CuArray(reshape(hcat([cholesky(Σ0s[i]).L for i in 1:N]...), Dx, Dx, N))
Qs_chol = CuArray(reshape(hcat([cholesky(Qs[i]).L for i in 1:N]...), Dx, Dx, N))
Rs_chol = CuArray(reshape(hcat([cholesky(Rs[i]).L for i in 1:N]...), Dy, Dy, N))

# Time update

# $$
# \mathbf{A}_2^{\top} \equiv\left[\begin{array}{cc}
# \sqrt{Q} & F \sqrt{P^{\circ}} \\
# \mathbf{0} & \sqrt{P^{\circ}}
# \end{array}\right], \quad \mathbf{B}_2^{\top} \equiv\left[\begin{array}{cc}
# \sqrt{P_{t+1}} & \mathbf{0} \\
# S \sqrt{P_{t+1}^*} & \sqrt{\Delta^{\circ}}
# \end{array}\right]
# $$

# Fill A using matrices
# TODO: use faster, filtering only method
A = CUDA.zeros(T, 2Dx, 2Dx, N)
A[1:Dx, 1:Dx, :] .= Qs_chol  # TODO: is this copying memory?
A[1:Dx, (Dx + 1):end, :] .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', As_gpu, Σ0s_chol)
A[(Dx + 1):end, (Dx + 1):end, :] .= Σ0s_chol

# Compute B via LQ demcomposition
B, _ = batch_lq(A)
Σ_pred_chol = B[1:Dx, 1:Dx, :]
# Σ_pred_chol = permutedims(Σ_pred_chol, (2, 1, 3))

# Update mean
μ_pred = CuArray{T}(undef, Dx, N)
CUDA.CUBLAS.gemv_strided_batched!('N', T(1.0), As_gpu, μ0s_gpu, T(0.0), μ_pred)
μ_pred .+= bs_gpu

# Compare to ground truth
As[1] * μ0s[1] .+ bs[1]
# println(As[1] * Σ0s[1] * As[1]' + Qs[1])
L = Array(Σ_pred_chol[:, :, 1])
Σ_pred = L * L'

println(As[1] * Σ0s[1] * As[1]' + Qs[1] - L * L')

# Measurement update

# $$
# \mathbf{A}_1^{\top} \equiv\left[\begin{array}{cc}
# \sqrt{R} & H \sqrt{P^{\sim}} \\
# \mathbf{0} & \sqrt{P^{\sim}}
# \end{array}\right], \quad \mathbf{B}_1^{\top} \equiv\left[\begin{array}{cc}
# \sqrt{Y} & \mathbf{0} \\
# K \sqrt{Y} & \sqrt{P^{\circ}}
# \end{array}\right]
# $$

# Fill A using matrices
A = CUDA.zeros(T, Dx + Dy, Dx + Dy, N)
A[1:Dy, 1:Dy, :] .= Rs_chol
A[1:Dy, (Dy + 1):end, :] .= CUDA.CUBLAS.gemm_strided_batched('N', 'N', Hs_gpu, Σ_pred_chol)
A[(Dy + 1):end, (Dy + 1):end, :] .= Σ_pred_chol

B, _ = batch_lq(A)
Σ_filt_chol = B[(Dy + 1):end, (Dy + 1):end, :]

# Compare to truth
L = Array(Σ_filt_chol[:, :, 1])
Σ_filt = L * L'
println(states[1].Σ - Σ_filt)

# Update state
sqrtY = B[1:Dy, 1:Dy, :]
KY = B[(Dy + 1):end, 1:Dy, :]

# sqrtY_ptrs = CUDA.CUBLAS.unsafe_strided_batch(sqrtY)

function prepare_batched_matrices(A::CuArray{T,3}) where {T}
    D1, D2, N = size(A)

    # Preallocate vector to hold the matrices
    matrices = Vector{CuMatrix{T}}(undef, N)

    # Create a view for each matrix in the batch
    for i in 1:N
        # Get a view of the i-th matrix
        matrices[i] = view(A, :, :, i)
    end

    return matrices
end

sqrtY_ptrs = prepare_batched_matrices(sqrtY)

Hx = CuArray{T}(undef, Dy, N)
CUDA.CUBLAS.gemv_strided_batched!('N', T(1.0), Hs_gpu, μ_pred, T(0.0), Hx)
y_res = CuArray(hcat(ys...)) - Hx .- cs_gpu

# Convert to matrices for trsm
y_res = reshape(y_res, Dy, 1, N)
y_res_ptrs = prepare_batched_matrices(y_res)

inv_term = CUDA.CUBLAS.trsm_batched('L', 'L', 'N', 'N', T(1.0), sqrtY_ptrs, y_res_ptrs)

function check_memory_layout(matrices::Vector{CuArray{T,2}}) where {T}
    # Get base pointers for each matrix
    ptrs = [Base.unsafe_convert(CuPtr{T}, m) for m in matrices]

    # Convert to actual addresses
    addresses = UInt.(ptrs)

    # Check if differences between consecutive addresses match matrix sizes
    expected_stride = sizeof(T) * prod(size(matrices[1]))
    actual_strides = diff(addresses)

    # Print diagnostics
    println("Expected stride between matrices: $expected_stride bytes")
    println("Actual strides: ", actual_strides)
    println("Contiguous: ", all(stride == expected_stride for stride in actual_strides))

    # Can also check if they're views of the same parent
    parent_ptrs = [pointer_from_objref(parent(m)) for m in matrices]
    same_parent = all(ptr == parent_ptrs[1] for ptr in parent_ptrs)
    return println("Same parent: $same_parent")
end

check_memory_layout(inv_term)

# Contiguous but don't have same parents so might be risky to use
# Let's just manually copy the data for now
inv_term_flat = CuArray{T}(undef, Dy, N)
for i in 1:N
    inv_term_flat[:, i] .= inv_term[i][:, 1]
end

product_term = CuArray{T}(undef, Dx, N)
CUDA.CUBLAS.gemv_strided_batched!('N', T(1.0), KY, inv_term_flat, T(0.0), product_term)
μ_filt = μ_pred .+ product_term

states[1].μ

println(Array(μ_filt[:, 1]) - states[1].μ)

# U = Array(Σ_filt_chol[:, :, 1])
# Σ_filt = U'U

# states[1].Σ
