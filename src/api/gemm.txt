#include "jla/jla.h"
#include "tensor/tensor_view.hpp"
#include "dispatch/gemm_dispatch.hpp"

jlaStatus_t jlaGemm(jlaHandle_t h,
                    jlaOp opA,
                    jlaOp opB,
                    float alpha,
                    jlaTensorView2D A,
                    jlaTensorView2D B,
                    float beta,
                    jlaTensorView2D C,
                    jlaGemmAlgo algo) {
    if (!h) return JLA_STATUS_INVALID_VALUE;

    if (A.dtype != B.dtype || A.dtype != C.dtype)
        return JLA_STATUS_INVALID_DTYPE;

    auto a = make_internal_view<float>(A);
    auto b = make_internal_view<float>(B);
    auto c = make_internal_view<float>(C);

    if (opA == JLA_OP_T) a = transpose_view(a);
    if (opB == JLA_OP_T) b = transpose_view(b);

    if (a.cols != b.rows ||
        c.rows != a.rows ||
        c.cols != b.cols)
        return JLA_STATUS_SHAPE_MISMATCH;

    GemmParams params {
        .A = a.data,
        .B = b.data,
        .C = c.data,
        .M = (int)a.rows,
        .N = (int)b.cols,
        .K = (int)a.cols,
        .lda = (int)a.ld,
        .ldb = (int)b.ld,
        .ldc = (int)c.ld,
        .alpha = alpha,
        .beta  = beta
    };

    GemmKernel kernel = pick_gemm_kernel(params, algo);
    if (!kernel)
        return JLA_STATUS_NOT_SUPPORTED;

    dim3 block(16, 16);
    dim3 grid((params.N + 15) / 16,
              (params.M + 15) / 16);

    kernel<<<grid, block, 0, h->stream>>>(params);
    return JLA_STATUS_SUCCESS;
}
