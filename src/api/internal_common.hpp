#pragma once
#include "jla/jla.h"
#include "tensor/tensor_view.hpp"
#include <cuda_runtime.h>

struct jlaHandle_ {
    cudaStream_t stream;
};
#define CUDA_CHECK(x) do {                                      
    cudaError_t err = (x);                                      
    if (err != cudaSuccess) {                                   
        std::cerr << "CUDA error: "                             
                  << cudaGetErrorString(err)                    
                  << " at " << __FILE__ << ":"                  
                  << __LINE__ << std::endl;                     
        std::exit(1);                                           
    }                                                           
} while (0)

#define CUBLAS_CHECK(x) do {                                    
    cublasStatus_t st = (x);                                    
    if (st != CUBLAS_STATUS_SUCCESS) {                          
        std::cerr << "cuBLAS error: " << (int)st                
                  << " at " << __FILE__ << ":"                  
                  << __LINE__ << std::endl;                     
        std::exit(1);                                           
    }                                                           
} while (0)