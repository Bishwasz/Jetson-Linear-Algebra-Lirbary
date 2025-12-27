#pragma once
#include "tensor/tensor_view.hpp"

// Just the kernel template. 
// jlaAdd will call this directly using <<< >>>.
template<typename T>
__global__ void elemwise_add_kernel(
    TensorView2D<T> A,
    TensorView2D<T> B,
    TensorView2D<T> C)
{
    // 1. Calculate linear index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A.rows * A.cols;

    // 2. Bounds check & Math
    if (idx < total) {
        // Optimization: Assumes ld == cols (Contiguous)
        C.data[idx] = A.data[idx] + B.data[idx];
    }
}