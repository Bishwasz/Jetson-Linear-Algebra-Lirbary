#pragma once
#include "tensor/tensor_view.hpp"

template<typename T>
__global__ void elemwise_add_kernel(
    TensorView2D<T> A,
    TensorView2D<T> B,
    TensorView2D<T> C)
{    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = A.rows * A.cols;

    if (idx < total) {
        C.data[idx] = A.data[idx] + B.data[idx];
    }
}