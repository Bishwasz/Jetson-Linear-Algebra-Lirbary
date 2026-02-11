#include "jla/jla.h"

#include <cuda_runtime.h>
#include <memory>
#include "internal_common.hpp"
#include "tensor/tensor_utils.hpp"

// 

jlaStatus_t jlaCreate(jlaHandle_t* out_handle) {
    if (!out_handle)
        return JLA_STATUS_INVALID_VALUE;

    auto handle = std::make_unique<jlaHandle_>();

    // Default: CUDA default stream (nullptr)
    handle->stream = nullptr;

    // Transfer ownership to C API
    *out_handle = handle.release();
    return JLA_STATUS_SUCCESS;
}

jlaStatus_t jlaDestroy(jlaHandle_t handle) {
    if (!handle)
        return JLA_STATUS_INVALID_VALUE;

    // Re-wrap raw pointer so RAII cleans it up
    std::unique_ptr<jlaHandle_> owned(handle);
    return JLA_STATUS_SUCCESS;
}

   //Stream Management

jlaStatus_t jlaSetStream(jlaHandle_t handle, void* cuda_stream) {
    if (!handle)
        return JLA_STATUS_INVALID_VALUE;

    handle->stream = static_cast<cudaStream_t>(cuda_stream);
    return JLA_STATUS_SUCCESS;
}

jlaStatus_t jlaGetStream(jlaHandle_t handle, void** out_stream) {
    if (!handle || !out_stream)
        return JLA_STATUS_INVALID_VALUE;

    *out_stream = reinterpret_cast<void*>(handle->stream);
    return JLA_STATUS_SUCCESS;
}
