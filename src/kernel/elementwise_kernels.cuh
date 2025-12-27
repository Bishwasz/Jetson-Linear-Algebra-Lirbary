#pragma once
#include "tensor/tensor_view.hpp"

/*
 * Note: These kernels currently assume CONTIGUOUS memory (ld == cols).
 * We treat the 2D tensors as flat 1D arrays for maximum performance.
 */

// 1. Addition: C = A + B
template<typename T>
__global__ void add_kernel(TensorView2D<T> A, TensorView2D<T> B, TensorView2D<T> C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A.rows * A.cols;

    if (idx < total) {
        C.data[idx] = A.data[idx] + B.data[idx];
    }
}

// 2. Subtraction: C = A - B
template<typename T>
__global__ void sub_kernel(TensorView2D<T> A, TensorView2D<T> B, TensorView2D<T> C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A.rows * A.cols;

    if (idx < total) {
        C.data[idx] = A.data[idx] - B.data[idx];
    }
}

// 3. Multiplication: C = A * B
template<typename T>
__global__ void mul_kernel(TensorView2D<T> A, TensorView2D<T> B, TensorView2D<T> C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A.rows * A.cols;

    if (idx < total) {
        C.data[idx] = A.data[idx] * B.data[idx];
    }
}