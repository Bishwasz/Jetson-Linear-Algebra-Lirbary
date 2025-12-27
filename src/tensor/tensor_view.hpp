#pragma once
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h> // <--- CRITICAL: Defines __host__ and __device__

/*
 * Internal, non-owning 2D tensor view used by kernels.
 * ...
 */
template <typename T>
struct TensorView2D {
    // This stops you from accidentally making a TensorView2D<void>
    static_assert(!std::is_void_v<T>, "TensorView2D<T>: T cannot be void");

    T* data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld   = 0;   // stride in elements

    // Using int64_t is great for safety, but remember that 
    // CUDA math is slightly faster with simple 'int' if 
    // you know your tensors are < 2GB.
    __host__ __device__ inline int64_t numel() const { return rows * cols; }

    // This check is very useful for assertions inside your API
    __host__ __device__ inline bool valid() const {
        return (data != nullptr) && (rows > 0) && (cols > 0) && (ld >= cols);
    }
};