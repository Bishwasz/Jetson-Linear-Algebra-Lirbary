#include "jla/jla.h"
#include "api/internal_common.hpp" // or your manual internal definitions
#include "tensor/tensor_utils.hpp"
#include <cuda_runtime.h>
#include "kernels/gemm_kernels.cuh"
#include <cuda_fp16.h>


jlaStatus_t jlaGemm(jlaHandle_t h, 
                    jlaOp opA, jlaOp opB,
                    float alpha, 
                    jlaTensorView2D A, 
                    jlaTensorView2D B, 
                    float beta, 
                    jlaTensorView2D C, 
                    jlaGemmAlgo algo) 
{
    if (!h) return JLA_STATUS_INVALID_VALUE;

    // 1. Basic Dimension Checks
    //    (Assumes opA=N, opB=N for simplicity as per requirements)
    if (A.cols != B.rows || A.rows != C.rows || B.cols != C.cols) {
        return JLA_STATUS_SHAPE_MISMATCH;
    }

    // 2. Algorithm Resolution (Handle 'AUTO')
    if (algo == JLA_GEMM_AUTO) {
        // If inputs are Half, default to TensorCores.
        if (A.dtype == JLA_F16 && B.dtype == JLA_F16) {
            algo = JLA_GEMM_TENSORCORE;
        } else {
            algo = JLA_GEMM_TILED;
        }
    }

    // 3. Dispatch based on Algorithm
    switch (algo) {
        case JLA_GEMM_TENSORCORE: {
            // Requirement: Inputs must be FP16 (half), Output must be FP32 (float)
            if (A.dtype != JLA_F16 || B.dtype != JLA_F16 || C.dtype != JLA_F32) {
                return JLA_STATUS_INVALID_DTYPE; 
            }

            // Create views with specific types <half> for inputs
            auto a_view = make_internal_view<half>(A);
            auto b_view = make_internal_view<half>(B);
            auto c_view = make_internal_view<float>(C);

            launch_sgemm_wmma(a_view, b_view, c_view, alpha, beta, h->stream);
            return JLA_STATUS_SUCCESS;
        }

        case JLA_GEMM_TILED:
        case JLA_GEMM_NAIVE: {
            // Requirement: All FP32
            if (A.dtype != JLA_F32 || B.dtype != JLA_F32 || C.dtype != JLA_F32) {
                return JLA_STATUS_INVALID_DTYPE;
            }

            // Create views with specific types <float> for everything
            auto a_view = make_internal_view<float>(A);
            auto b_view = make_internal_view<float>(B);
            auto c_view = make_internal_view<float>(C);

            launch_gemm(a_view, b_view, c_view, alpha, beta, h->stream);
            return JLA_STATUS_SUCCESS;
        }

        default:
            return JLA_STATUS_NOT_SUPPORTED;
    }
}