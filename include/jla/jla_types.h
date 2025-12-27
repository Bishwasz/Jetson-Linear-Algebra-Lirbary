#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

//   Status / Error Codes
typedef enum {
    JLA_STATUS_SUCCESS = 0,
    JLA_STATUS_INVALID_VALUE,
    JLA_STATUS_INVALID_LAYOUT,
    JLA_STATUS_INVALID_DTYPE,
    JLA_STATUS_SHAPE_MISMATCH,
    JLA_STATUS_ALLOC_FAILED,
    JLA_STATUS_NOT_SUPPORTED,
    JLA_STATUS_INTERNAL_ERROR
} jlaStatus_t;
// Data Types
 
typedef enum {
    JLA_F16 = 0,
    JLA_F32 = 1
} jlaDType;

//   Memory Layout

typedef enum {
    JLA_ROW_MAJOR = 0,
    JLA_COL_MAJOR = 1
} jlaLayout;

// Operation Flags (GEMM)

typedef enum {
    JLA_OP_N = 0,   /* No transpose */
    JLA_OP_T = 1    /* Transpose */
} jlaOp;

/* ================================
   GEMM Algorithm Selection
   ================================ */

typedef enum {
    JLA_GEMM_AUTO = 0,     /* Heuristic-based selection */
    JLA_GEMM_NAIVE,
    JLA_GEMM_TILED,
    JLA_GEMM_TENSORCORE
} jlaGemmAlgo;

/* ================================
   Tensor View (2D)
   ================================ */

/*
 * NOTE:
 *  - This is a NON-OWNING view.
 *  - `data` must be a device pointer (CUDA).
 *  - `ld` is the leading dimension (stride).
 *  - Supports submatrices and transposed views.
 */
typedef struct {
    void*      data;
    int64_t    rows;
    int64_t    cols;
    int64_t    ld;        /* leading dimension */
    jlaDType   dtype;
    jlaLayout  layout;
} jlaTensorView2D;

/* ================================
   Handle (opaque)
   ================================ */

typedef struct jlaHandle_* jlaHandle_t;

#ifdef __cplusplus
}
#endif
