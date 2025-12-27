#include "jla/jla.h"
#include "tensor/tensor_view.hpp"
#include "kernels/transpose.cuh"

jlaStatus_t jlaTranspose(jlaHandle_t h,
                         jlaTensorView2D A,
                         jlaTensorView2D AT) {
    if (!h) return JLA_STATUS_INVALID_VALUE;

    if (AT.rows != A.cols || AT.cols != A.rows)
        return JLA_STATUS_SHAPE_MISMATCH;

    if (A.dtype != AT.dtype)
        return JLA_STATUS_INVALID_DTYPE;

    auto a  = make_internal_view<float>(A);
    auto at = make_internal_view<float>(AT);

    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((A.cols + TILE - 1) / TILE,
              (A.rows + TILE - 1) / TILE);

    transpose_kernel<<<grid, block, 0, h->stream>>>(a, at);
    return JLA_STATUS_SUCCESS;
}
