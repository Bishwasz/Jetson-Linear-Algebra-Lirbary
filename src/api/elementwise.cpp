#include "jla/jla.h"
#include "kernels/elementwise.h" // Include the bridge header
#include "utils/checks.hpp"

// --- Public API Implementation ---

jlaStatus_t jlaAdd(jlaHandle_t h, jlaTensorView2D A, jlaTensorView2D B, jlaTensorView2D C) {
    if (!h) return JLA_STATUS_INVALID_VALUE;
    if (!same_shape(A, B, C)) return JLA_STATUS_SHAPE_MISMATCH;
    if (!same_dtype(A, B, C)) return JLA_STATUS_INVALID_DTYPE;

    // Convert opaque C structs to Internal C++ Views
    auto a = make_internal_view<float>(A);
    auto b = make_internal_view<float>(B);
    auto c = make_internal_view<float>(C);

    // DELEGATE to the launcher
    // No <<<grid, block>>> calculation here!
    launch_add(a, b, c, h->stream);

    return JLA_STATUS_SUCCESS;
}

jlaStatus_t jlaSub(jlaHandle_t h, jlaTensorView2D A, jlaTensorView2D B, jlaTensorView2D C) {
    if (!h) return JLA_STATUS_INVALID_VALUE;
    if (!same_shape(A, B, C)) return JLA_STATUS_SHAPE_MISMATCH;
    if (!same_dtype(A, B, C)) return JLA_STATUS_INVALID_DTYPE;

    auto a = make_internal_view<float>(A);
    auto b = make_internal_view<float>(B);
    auto c = make_internal_view<float>(C);

    launch_sub(a, b, c, h->stream);

    return JLA_STATUS_SUCCESS;
}

jlaStatus_t jlaElemMul(jlaHandle_t h, jlaTensorView2D A, jlaTensorView2D B, jlaTensorView2D C) {
    if (!h) return JLA_STATUS_INVALID_VALUE;
    if (!same_shape(A, B, C)) return JLA_STATUS_SHAPE_MISMATCH;
    if (!same_dtype(A, B, C)) return JLA_STATUS_INVALID_DTYPE;

    auto a = make_internal_view<float>(A);
    auto b = make_internal_view<float>(B);
    auto c = make_internal_view<float>(C);

    launch_mul(a, b, c, h->stream);

    return JLA_STATUS_SUCCESS;
}