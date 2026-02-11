cmake_minimum_required(VERSION 3.18)

project(JetsonLinearAlgebra LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# 1. FIND CUDA TOOLKIT (Fixes "cuda_runtime.h not found")
find_package(CUDAToolkit REQUIRED)  # <--- NEW LINE

# 2. Include Directories
include_directories(include src)

# 3. Gather Source Files
file(GLOB API_SOURCES "src/api/*.cpp")
file(GLOB KERNEL_SOURCES "src/kernels/*.cu")

# 4. Create the Shared Library
add_library(jla SHARED 
    ${API_SOURCES} 
    ${KERNEL_SOURCES}
)

# 5. LINK CUDA RUNTIME (Fixes linker errors later)
# This automatically adds the include paths for cuda_runtime.h
target_link_libraries(jla PRIVATE CUDA::cudart) # <--- NEW LINE

set_property(TARGET jla PROPERTY CUDA_ARCHITECTURES 87)
set_property(TARGET jla PROPERTY POSITION_INDEPENDENT_CODE ON)


# --- Test Application 1: Elementwise ---
# Renamed from 'test_jla' to 'test_elementwise' for clarity
add_executable(test_elementwise tests/test_elementwise.cpp)
target_link_libraries(test_elementwise PRIVATE jla CUDA::cudart)

# --- Test Application 2: GEMM ---
# This needs its own unique name
add_executable(test_gemm tests/test_gemm.cpp)
target_link_libraries(test_gemm PRIVATE jla CUDA::cudart)

add_executable(test_tensorcore tests/test_tensorcore.cu)
target_link_libraries(test_tensorcore
    PRIVATE
        jla
        CUDA::cudart
        CUDA::cublas
)