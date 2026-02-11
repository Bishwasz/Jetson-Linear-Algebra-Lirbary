#pragma once
#include <cstdint>
#include <type_traits>
#include <cuda_runtime.h> 


template <typename T>
struct TensorView2D {
    static_assert(!std::is_void_v<T>, "TensorView2D<T>: T cannot be void");

    T* data = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t ld   = 0;   


    __host__ __device__ inline int64_t numel() const { return rows * cols; }

    __host__ __device__ inline bool valid() const {
        return (data != nullptr) && (rows > 0) && (cols > 0) && (ld >= cols);
    }
};