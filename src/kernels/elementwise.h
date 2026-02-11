#pragma once
#include "tensor/tensor_view.hpp"
#include <cuda_runtime.h>
void launch_add(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream);
void launch_sub(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream);
void launch_mul(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream);