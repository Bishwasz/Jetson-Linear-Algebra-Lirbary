#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "tensor/tensor_view.hpp"

void launch_gemm(
    TensorView2D<float> A,
    TensorView2D<float> B,
    TensorView2D<float> C,
    float alpha,
    float beta,
    cudaStream_t stream
);

void launch_sgemm_wmma(
    TensorView2D<half>  A,
    TensorView2D<half>  B,
    TensorView2D<float> C,
    float alpha,
    float beta,
    cudaStream_t stream
);
