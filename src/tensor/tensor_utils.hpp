// src/tensor/tensor_utils.hpp
#pragma once
#include <cstdint>

#include "jla/jla_types.h"
#include "tensor_view.hpp"

/*
 * Utilities for converting public API tensor views (jlaTensorView2D)
 * into internal kernel-friendly TensorView2D<T>, and for creating
 * zero-copy transformed "views" (transpose, submatrix).
 *
 * NOTE:
 *  - These do NOT allocate memory.
 *  - These do NOT move/copy data.
 *  - They only adjust metadata (shape/ld/pointer offset).
 */

/* ----------------------------
   DType helpers
   ---------------------------- */

inline bool jla_dtype_matches_float(jlaDType dt) {
    return dt == JLA_F32;
}

/* ----------------------------
   Conversion: public -> internal
   ---------------------------- */

template <typename T>
inline TensorView2D<T> make_internal_view(const jlaTensorView2D& t) {
    return TensorView2D<T>{
        static_cast<T*>(t.data),
        t.rows,
        t.cols,
        t.ld
    };
}

/*
 * Like make_internal_view, but normalizes layout so kernels can assume
 * row-major semantics.
 *
 * Current policy:
 *  - If public view is row-major: return as-is.
 *  - If public view is col-major: reinterpret it as the transpose of a row-major view.
 *
 * This works because column-major storage for (rows, cols, ld)
 * is equivalent to a row-major view of the transposed matrix with the same ld.
 */
template <typename T>
inline TensorView2D<T> normalize_layout(const jlaTensorView2D& t) {
    TensorView2D<T> v = make_internal_view<T>(t);
    if (t.layout == JLA_COL_MAJOR) {
        // reinterpret as transpose view
        return TensorView2D<T>{ v.data, v.cols, v.rows, v.ld };
    }
    return v; // row-major
}

/* ----------------------------
   Zero-copy transpose view
   ---------------------------- */

/*
 * Logical transpose (zero-copy):
 *  - same data pointer
 *  - swapped rows/cols
 *  - same ld (stride in elements)
 *
 * IMPORTANT:
 *  - This is valid as a view, but your kernel must interpret indexing
 *    consistently (i*ld + j).
 */
template <typename T>
inline TensorView2D<T> transpose_view(const TensorView2D<T>& t) {
    return TensorView2D<T>{ t.data, t.cols, t.rows, t.ld };
}

/* ----------------------------
   Submatrix view (zero-copy)
   ---------------------------- */

/*
 * Returns a view of a submatrix starting at (r0, c0) with size (r, c).
 * No bounds checks are performed here (do that in API layer if desired).
 */
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
