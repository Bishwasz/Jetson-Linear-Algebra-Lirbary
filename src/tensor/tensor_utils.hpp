// src/tensor/tensor_utils.hpp
#pragma once
#include <cstdint>

#include "jla/jla_types.h"
#include "tensor_view.hpp"



inline bool jla_dtype_matches_float(jlaDType dt) {
    return dt == JLA_F32;
}


template <typename T>
inline TensorView2D<T> make_internal_view(const jlaTensorView2D& t) {
    return TensorView2D<T>{
        static_cast<T*>(t.data),
        t.rows,
        t.cols,
        t.ld
    };
}
e <typename T>
inline TensorView2D<T> normalize_layout(const jlaTensorView2D& t) {
    TensorView2D<T> v = make_internal_view<T>(t);
    if (t.layout == JLA_COL_MAJOR) {
        // reinterpret as transpose view
        return TensorView2D<T>{ v.data, v.cols, v.rows, v.ld };
    }
    return v; // row-major
}


template <typename T>
inline TensorView2D<T> transpose_view(const TensorView2D<T>& t) {
    return TensorView2D<T>{ t.data, t.cols, t.rows, t.ld };
}

template <typename T>
inline TensorView2D<T> submatrix_view(const TensorView2D<T>& t,
                                      int64_t r0, int64_t c0,
                                      int64_t r,  int64_t c) {
    return TensorView2D<T>{
        t.data + r0 * t.ld + c0,
        r,
        c,
        t.ld
    };
}
