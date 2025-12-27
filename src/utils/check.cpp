#pragma once
#include "jla/jla_types.h"

inline bool same_shape(const jlaTensorView2D& A,
                       const jlaTensorView2D& B,
                       const jlaTensorView2D& C) {
    return (A.rows == B.rows) &&
           (A.cols == B.cols) &&
           (A.rows == C.rows) &&
           (A.cols == C.cols);
}
inline bool same_dtype(const jlaTensorView2D& A,
                       const jlaTensorView2D& B,
                       const jlaTensorView2D& C) {
    return (A.dtype == B.dtype) &&
           (A.dtype == C.dtype);
}

inline bool valid_tensor(const jlaTensorView2D& T) {
    return T.data != nullptr &&
           T.rows > 0 &&
           T.cols > 0 &&
           T.ld >= T.cols;
}
