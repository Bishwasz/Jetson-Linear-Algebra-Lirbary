#include "elementwise.h"        // Implements the header
#include "elementwise_kernels.cuh" // Includes your __global__ templates

static void get_grid(int total, dim3& grid, dim3& block) {
    block.x = 256;
    grid.x = (total + block.x - 1) / block.x;
}

void launch_add(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream) {
    dim3 grid, block;
    get_grid(A.rows * A.cols, grid, block);
    add_kernel<float><<<grid, block, 0, stream>>>(A, B, C);
}

void launch_sub(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream) {
    dim3 grid, block;
    get_grid(A.rows * A.cols, grid, block);
    sub_kernel<float><<<grid, block, 0, stream>>>(A, B, C);
}

void launch_mul(TensorView2D<float> A, TensorView2D<float> B, TensorView2D<float> C, cudaStream_t stream) {
    dim3 grid, block;
    get_grid(A.rows * A.cols, grid, block);
    mul_kernel<float><<<grid, block, 0, stream>>>(A, B, C);
}