#pragma once
#include "jla_types.h"
#include "jla_types.h"
#ifdef __cplusplus
extern "C" {
#endif


/*
 * Create a library handle.
 * The handle stores execution context (stream, device state).
 */
jlaStatus_t jlaCreate(jlaHandle_t* out_handle);
jlaStatus_t jlaDestroy(jlaHandle_t handle);

/*
 * Associate a CUDA stream with the handle.
 * The stream is passed as an opaque pointer to avoid CUDA headers.
 */
jlaStatus_t jlaSetStream(jlaHandle_t handle, void* cuda_stream);
jlaStatus_t jlaGetStream(jlaHandle_t handle, void** out_stream);


 jlaStatus_t jlaAdd(jlaHandle_t handle,
                   jlaTensorView2D A,
                   jlaTensorView2D B,
                   jlaTensorView2D C);
jlaStatus_t jlaSub(jlaHandle_t handle,
                   jlaTensorView2D A,
                   jlaTensorView2D B,
                   jlaTensorView2D C);
jlaStatus_t jlaElemMul(jlaHandle_t handle,
                       jlaTensorView2D A,
                       jlaTensorView2D B,
                       jlaTensorView2D C);

/* =========================================================
   Matrix Transpose
   ========================================================= */

/*
 * Matrix transpose: AT = transpose(A)
 *
 * Requirements:
 *  - AT.rows == A.cols
 *  - AT.cols == A.rows
 */
jlaStatus_t jlaTranspose(jlaHandle_t handle,
                          jlaTensorView2D A,
                          jlaTensorView2D AT);
jlaStatus_t jlaGemm(jlaHandle_t handle,
                    jlaOp opA,
                    jlaOp opB,
                    float alpha,
                    jlaTensorView2D A,
                    jlaTensorView2D B,
                    float beta,
                    jlaTensorView2D C,
                    jlaGemmAlgo algo);
const char* jlaGetStatusString(jlaStatus_t status);
const char* jlaGetVersion(void);

#ifdef __cplusplus
}
#endif
