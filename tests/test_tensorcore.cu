#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "jla/jla.h"
#include "jla/jla_types.h"
void cpu_gemm(int M, int N, int K,
              float alpha, const float* A, const float* B,
              float beta, float* C)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

// ------------------------------------------------------------
// FP32 -> FP16 conversion kernel
// ------------------------------------------------------------
__global__ void float_to_half_kernel(const float* src, half* dst, int count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) dst[idx] = __float2half(src[idx]);
}

void convert_to_half(const float* d_src, half* d_dst, int count, cudaStream_t stream = 0)
{
    int blockSize = 256;
    int gridSize = (count + blockSize - 1) / blockSize;
    float_to_half_kernel<<<gridSize, blockSize, 0, stream>>>(d_src, d_dst, count);
    CUDA_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------
// cuBLAS FP32 benchmark (NOTE: cuBLAS assumes column-major by default)
// If your data is row-major, compute C = A*B by calling cuBLAS on swapped
// operands: C^T = B^T * A^T  => treat row-major buffers as column-major.
// This trick is standard for fair benchmarking without transposes.
// ------------------------------------------------------------
float benchmark_cublas_sgemm_rowmajor(
    cublasHandle_t handle,
    cudaStream_t stream,
    int M, int N, int K,
    const float* d_A_rowmajor, // MxK row-major
    const float* d_B_rowmajor, // KxN row-major
    float* d_C_rowmajor,       // MxN row-major
    int iters = 100)
{
    // In column-major terms:
    // Row-major A (MxK) == Col-major A^T (KxM)
    // Row-major B (KxN) == Col-major B^T (NxK)
    // Row-major C (MxN) == Col-major C^T (NxM)
    //
    // So compute C^T (NxM) = B^T (NxK) * A^T (KxM)
    // => cuBLAS GEMM: (N x K) * (K x M) = (N x M)

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // Leading dims for column-major views:
    // B^T is (N x K) col-major -> leading dim = N
    // A^T is (K x M) col-major -> leading dim = K
    // C^T is (N x M) col-major -> leading dim = N
    int ldb = N;
    int lda = K;
    int ldc = N;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Warm-up
    for (int i = 0; i < 10; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/N, /*n=*/M, /*k=*/K,
            &alpha,
            d_B_rowmajor, ldb,
            d_A_rowmajor, lda,
            &beta,
            d_C_rowmajor, ldc
        ));
    }

    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            /*m=*/N, /*n=*/M, /*k=*/K,
            &alpha,
            d_B_rowmajor, ldb,
            d_A_rowmajor, lda,
            &beta,
            d_C_rowmajor, ldc
        ));
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return ms;
}

static inline double gflops_gemm(int M, int N, int K, double ms)
{
    // 2*M*N*K floating ops
    return (2.0 * (double)M * (double)N * (double)K) / (ms * 1e6);
}

// ------------------------------------------------------------
// JLA GEMM benchmark (Tensor Core path)
// ------------------------------------------------------------
double benchmark_jla_gemm_tensorcore(
    jlaHandle_t handle,
    int M, int N, int K,
    float alpha,
    const jlaTensorView2D& A,
    const jlaTensorView2D& B,
    float beta,
    const jlaTensorView2D& C,
    int iters = 50
) {
    // Reset C (assumes row-major contiguous for this benchmark)
    CUDA_CHECK(cudaMemset(C.data, 0, (size_t)M * (size_t)N * sizeof(float)));

    // Warm-up
    jlaGemm(handle, JLA_OP_N, JLA_OP_N, alpha, A, B, beta, C, JLA_GEMM_TENSORCORE);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        jlaGemm(handle, JLA_OP_N, JLA_OP_N, alpha, A, B, beta, C, JLA_GEMM_TENSORCORE);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    double avg_ms = elapsed_ms / iters;
    return avg_ms;
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------
int main()
{
    int M = 4096, N = 4096, K = 4096;
    float alpha = 1.0f;
    float beta  = 0.0f;

    std::cout << "=== JLA vs cuBLAS GEMM Benchmark ===\n";
    std::cout << "M=" << M << " N=" << N << " K=" << K << "\n";

    size_t elements_A = (size_t)M * (size_t)K;
    size_t elements_B = (size_t)K * (size_t)N;
    size_t elements_C = (size_t)M * (size_t)N;

    // Host buffers
    std::vector<float> h_A(elements_A);
    std::vector<float> h_B(elements_B);
    std::vector<float> h_C(elements_C, 0.0f);
    std::vector<float> h_C_ref(elements_C, 0.0f);

    for (auto& v : h_A) v = (float)std::rand() / RAND_MAX * 0.5f;
    for (auto& v : h_B) v = (float)std::rand() / RAND_MAX * 0.5f;

    // Device buffers
    float *d_A_f32 = nullptr, *d_B_f32 = nullptr, *d_C = nullptr;
    half  *d_A_f16 = nullptr, *d_B_f16 = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A_f32, elements_A * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B_f32, elements_B * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_A_f16, elements_A * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_B_f16, elements_B * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_C,     elements_C * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_A_f32, h_A.data(), elements_A * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B_f32, h_B.data(), elements_B * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_C, 0, elements_C * sizeof(float)));

    // cuBLAS setup
    cublasHandle_t blas;
    CUBLAS_CHECK(cublasCreate(&blas));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(blas, stream));

    // Convert FP32 -> FP16 for JLA tensorcore path (on same stream)
    convert_to_half(d_A_f32, d_A_f16, (int)elements_A, stream);
    convert_to_half(d_B_f32, d_B_f16, (int)elements_B, stream);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // JLA setup
    jlaHandle_t jla;
    jlaCreate(&jla);

    jlaTensorView2D view_A = { d_A_f16, M, K, K, JLA_F16, JLA_ROW_MAJOR };
    jlaTensorView2D view_B = { d_B_f16, K, N, N, JLA_F16, JLA_ROW_MAJOR };
    jlaTensorView2D view_C = { d_C,     M, N, N, JLA_F32, JLA_ROW_MAJOR };

    // Sanity call
    jlaStatus_t st = jlaGemm(jla, JLA_OP_N, JLA_OP_N, alpha, view_A, view_B, beta, view_C, JLA_GEMM_TENSORCORE);
    if (st != JLA_STATUS_SUCCESS) {
        std::cerr << "JLA GEMM failed with code " << st << "\n";
        return -1;
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark JLA
    double jla_ms = benchmark_jla_gemm_tensorcore(jla, M, N, K, alpha, view_A, view_B, beta, view_C, 50);
    double jla_gflops = gflops_gemm(M, N, K, jla_ms);
    std::cout << "[JLA TensorCore]  " << jla_ms << " ms, " << jla_gflops << " GFLOP/s\n";
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C,
                      elements_C * sizeof(float),
                      cudaMemcpyDeviceToHost));

double jla_checksum = 0.0;
for (int i = 0; i < 100; ++i)
    jla_checksum += h_C[i];

std::cout << "[JLA]    checksum(first100) = " << jla_checksum << "\n";

    // Benchmark cuBLAS FP32 (row-major trick)
    CUDA_CHECK(cudaMemsetAsync(d_C, 0, elements_C * sizeof(float), stream));
    float blas_ms = benchmark_cublas_sgemm_rowmajor(blas, stream, M, N, K, d_A_f32, d_B_f32, d_C, 100);
    double blas_gflops = gflops_gemm(M, N, K, (double)blas_ms);
    std::cout << "[cuBLAS FP32]     " << blas_ms << " ms, " << blas_gflops << " GFLOP/s\n";

    // Copy back cuBLAS result for verification
    CUDA_CHECK(cudaMemcpyAsync(h_C.data(), d_C, elements_C * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // CPU reference (slow for 4096^3; skip if you want)
    // NOTE: This is huge (4096^3) on CPU; leaving here but you may want smaller sizes for verify.
    // cpu_gemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());

    // Quick correctness spot-check (optional): compare against CPU only if you compute it.
    // For now, just print a small checksum-like statistic.
    double checksum = 0.0;
    for (int i = 0; i < 100; ++i) checksum += h_C[i];
    std::cout << "[cuBLAS] checksum(first100) = " << checksum << "\n";

    // Cleanup
    jlaDestroy(jla);
    CUBLAS_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(d_A_f32));
    CUDA_CHECK(cudaFree(d_B_f32));
    CUDA_CHECK(cudaFree(d_A_f16));
    CUDA_CHECK(cudaFree(d_B_f16));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
