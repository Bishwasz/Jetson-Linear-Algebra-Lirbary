#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include "jla/jla.h"

// --- Error Checking Helpers ---
#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

#define CHECK_JLA(call) { \
    jlaStatus_t err = call; \
    if (err != JLA_STATUS_SUCCESS) { \
        std::cerr << "JLA Error code: " << err << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

int main() {
    std::cout << "--- Starting JLA Element-wise Test ---" << std::endl;

    // 1. Setup Dimensions (1024x1024 matrix)
    int rows = 1024;
    int cols = 1024;
    int n = rows * cols;
    size_t bytes = n * sizeof(float);

    // 2. Allocate Host Memory (CPU)
    std::vector<float> h_A(n);
    std::vector<float> h_B(n);
    std::vector<float> h_C(n);

    // Initialize Data (A = 1.0, B = 2.0)
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 3. Allocate Device Memory (GPU)
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, bytes));
    CHECK_CUDA(cudaMalloc(&d_B, bytes));
    CHECK_CUDA(cudaMalloc(&d_C, bytes));

    // 4. Copy Data: CPU -> GPU
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    // 5. Initialize Library Handle
    jlaHandle_t handle;
    CHECK_JLA(jlaCreate(&handle));

    // 6. Define Tensor Views (Simple Contiguous)
    // ld (stride) is equal to cols because it's packed
    jlaTensorView2D viewA = {d_A, rows, cols, cols, JLA_FLOAT32};
    jlaTensorView2D viewB = {d_B, rows, cols, cols, JLA_FLOAT32};
    jlaTensorView2D viewC = {d_C, rows, cols, cols, JLA_FLOAT32};

    // ==========================================
    // TEST 1: ADDITION (1.0 + 2.0 = 3.0)
    // ==========================================
    std::cout << "Running jlaAdd..." << std::endl;
    CHECK_JLA(jlaAdd(handle, viewA, viewB, viewC));

    // Copy Result Back: GPU -> CPU
    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify Result
    float error = 0.0f;
    for (int i = 0; i < n; i++) {
        error += std::abs(h_C[i] - 3.0f);
    }
    if (error < 0.01f) std::cout << "✅ jlaAdd Passed!" << std::endl;
    else std::cout << "❌ jlaAdd Failed! Error sum: " << error << std::endl;


    // ==========================================
    // TEST 2: MULTIPLICATION (1.0 * 2.0 = 2.0)
    // ==========================================
    std::cout << "Running jlaElemMul..." << std::endl;
    CHECK_JLA(jlaElemMul(handle, viewA, viewB, viewC));

    CHECK_CUDA(cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    error = 0.0f;
    for (int i = 0; i < n; i++) {
        error += std::abs(h_C[i] - 2.0f);
    }
    if (error < 0.01f) std::cout << "✅ jlaElemMul Passed!" << std::endl;
    else std::cout << "❌ jlaElemMul Failed! Error sum: " << error << std::endl;

    // 7. Cleanup
    CHECK_JLA(jlaDestroy(handle));
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}