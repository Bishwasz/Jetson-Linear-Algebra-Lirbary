#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "jla/jla.h"
#include "jla/jla_types.h"

// CPU Reference implementation for verification
void cpu_gemm(int M, int N, int K, 
              float alpha, const float* A, const float* B, 
              float beta, float* C) 
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                // Row-major standard logic: A[i][k] * B[k][j]
                sum += A[i * K + k] * B[k * N + j];
            }
            // C = alpha*(A*B) + beta*C
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main() {
    // 1. Setup Dimensions
    // Use non-square dimensions to catch shape bugs!
    int M = 64; 
    int N = 128;
    int K = 256;

    float alpha = 1.0f;
    float beta  = 0.0f; // Pure matmul, overwrite C

    std::cout << "Testing GEMM with M=" << M << ", N=" << N << ", K=" << K << "...\n";

    // 2. Allocate Host Memory
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);      // GPU result destination
    std::vector<float> h_C_ref(M * N, 0.0f);  // CPU reference

    // Initialize random data
    for (auto& v : h_A) v = static_cast<float>(rand()) / RAND_MAX;
    for (auto& v : h_B) v = static_cast<float>(rand()) / RAND_MAX;

    // 3. Allocate Device Memory
    void *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A.data(), size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), size_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C.data(), size_C, cudaMemcpyHostToDevice); // Init C with zeros/beta data

    // 4. Initialize JLA Library
    jlaHandle_t handle;
    jlaCreate(&handle);

    // Create Views
    jlaTensorView2D view_A = { d_A, M, K, K, JLA_F32, JLA_ROW_MAJOR };
    jlaTensorView2D view_B = { d_B, K, N, N, JLA_F32, JLA_ROW_MAJOR };
    jlaTensorView2D view_C = { d_C, M, N, N, JLA_F32, JLA_ROW_MAJOR };

    // 5. Run GPU GEMM
    std::cout << "Running GPU Kernel...\n";
    jlaStatus_t status = jlaGemm(handle, 
                                 JLA_OP_N, JLA_OP_N, // No Transpose
                                 alpha, 
                                 view_A, view_B, 
                                 beta, 
                                 view_C, JLA_GEMM_AUTO);

    if (status != JLA_STATUS_SUCCESS) {
        std::cerr << "JLA GEMM Failed with status: " << status << std::endl;
        return -1;
    }

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // 6. Verify Results
    // Get GPU result back
    cudaMemcpy(h_C.data(), d_C, size_C, cudaMemcpyDeviceToHost);

    // Calculate CPU Reference
    std::cout << "Calculating CPU Reference...\n";
    cpu_gemm(M, N, K, alpha, h_A.data(), h_B.data(), beta, h_C_ref.data());

    // Compare
    float max_error = 0.0f;
    for (size_t i = 0; i < h_C.size(); ++i) {
        float diff = std::abs(h_C[i] - h_C_ref[i]);
        if (diff > max_error) max_error = diff;
    }

    std::cout << "Max Error: " << max_error << std::endl;

    if (max_error < 1e-4) {
        std::cout << "SUCCESS: GPU matches CPU!\n";
    } else {
        std::cout << "FAILURE: Results mismatch!\n";
    }

    // 7. Cleanup
    jlaDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}