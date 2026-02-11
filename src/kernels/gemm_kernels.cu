#include "gemm_kernels.cuh"
#include <mma.h>
#include <cuda_fp16.h>

using namespace nvcuda;

// CONFIGURATION
const int BLOCKSIZE = 128;
// Tile Dimensions
const int BM = 64; // Block M
const int BN = 64; // Block N
const int BK = 16; // Block K

// Warp Dimensions
const int WM = 32;
const int WN = 32;

const int PAD = 8; 

// 1. OPTIMIZED TENSOR CORE KERNEL (TILED)
__global__ void sgemm_wmma_tiled_kernel(
    TensorView2D<half>  A,
    TensorView2D<half>  B,
    TensorView2D<float> C,
    float alpha,
    float beta
) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.x;

    int row_offset = by * BM;
    int col_offset = bx * BN;

    // Double buffered shared memory
    __shared__ half s_A[2][BM * (BK + PAD)];
    __shared__ half s_B[2][BK * (BN + PAD)];

    // Fragments
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;

    // Initialize accumulators
    #pragma unroll
    for(int i=0; i<2; ++i) {
        #pragma unroll
        for(int j=0; j<2; ++j) {
            wmma::fill_fragment(acc[i][j], 0.0f);
        }
    }

    int K = A.cols;
    int write_idx = 0;
    int read_idx = 0;

    // ============================================
    // PREFETCH FIRST TILE (k=0) into buffer 0
    // ============================================
    {
        int k = 0;
        int t_idx = tid;
        
        // Load A tile
        while (t_idx < BM * BK) {
            int row = t_idx / BK;
            int col = t_idx % BK;
            
            half val;
            if (row_offset + row < A.rows && k + col < A.cols) {
                val = A.data[(row_offset + row) * A.ld + (k + col)];
            } else {
                val = __float2half(0.0f);
            }
            
            s_A[write_idx][row * (BK + PAD) + col] = val;
            t_idx += blockDim.x;
        }

        // Load B tile
        t_idx = tid;
        while (t_idx < BK * BN) {
            int row = t_idx / BN;
            int col = t_idx % BN;
            
            half val;
            if (k + row < B.rows && col_offset + col < B.cols) {
                val = B.data[(k + row) * B.ld + (col_offset + col)];
            } else {
                val = __float2half(0.0f);
            }
            
            s_B[write_idx][row * (BN + PAD) + col] = val;
            t_idx += blockDim.x;
        }
    }
    
    __syncthreads();

    // MAIN LOOP - Start from k=BK (second tile)
    for (int k = BK; k <= K; k += BK) {
        
        read_idx = write_idx;
        write_idx = 1 - write_idx;
        
        if (k < K) {
            int t_idx = tid;
            
            while (t_idx < BM * BK) {
                int row = t_idx / BK;
                int col = t_idx % BK;
                
                half val;
                if (row_offset + row < A.rows && k + col < A.cols) {
                    val = A.data[(row_offset + row) * A.ld + (k + col)];
                } else {
                    val = __float2half(0.0f);
                }
                
                s_A[write_idx][row * (BK + PAD) + col] = val;
                t_idx += blockDim.x;
            }

            // Load B tile
            t_idx = tid;
            while (t_idx < BK * BN) {
                int row = t_idx / BN;
                int col = t_idx % BN;
                
                half val;
                if (k + row < B.rows && col_offset + col < B.cols) {
                    val = B.data[(k + row) * B.ld + (col_offset + col)];
                } else {
                    val = __float2half(0.0f);
                }
                
                s_B[write_idx][row * (BN + PAD) + col] = val;
                t_idx += blockDim.x;
            }
        }

        int warpId = tid / 32;
        int warpRow = warpId / 2; 
        int warpCol = warpId % 2; 

        #pragma unroll
        for (int i = 0; i < 2; ++i) {     
            #pragma unroll
            for (int j = 0; j < 2; ++j) { 
                
                int m_idx = warpRow * WM + i * 16;
                int n_idx = warpCol * WN + j * 16;

                const half* a_ptr = &s_A[read_idx][m_idx * (BK + PAD) + 0];
                const half* b_ptr = &s_B[read_idx][0 * (BN + PAD) + n_idx];

                wmma::load_matrix_sync(a_frag, a_ptr, BK + PAD);
                wmma::load_matrix_sync(b_frag, b_ptr, BN + PAD);

                wmma::mma_sync(acc[i][j], a_frag, b_frag, acc[i][j]);
            }
        }
        
        __syncthreads();
    }

    int warpId = tid / 32;
    int warpRow = warpId / 2;
    int warpCol = warpId % 2;

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            int globalRow = row_offset + warpRow * WM + i * 16;
            int globalCol = col_offset + warpCol * WN + j * 16;

            if (globalRow < C.rows && globalCol < C.cols) {
                float* c_ptr = C.data + globalRow * C.ld + globalCol;

                #pragma unroll
                for(int t=0; t<acc[i][j].num_elements; t++) {
                    acc[i][j].x[t] *= alpha;
                }

                if (beta != 0.0f) {
                    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
                    wmma::load_matrix_sync(c_frag, c_ptr, C.ld, wmma::mem_row_major);
                    
                    #pragma unroll
                    for(int t=0; t<c_frag.num_elements; t++) {
                        acc[i][j].x[t] += beta * c_frag.x[t];
                    }
                }

                wmma::store_matrix_sync(c_ptr, acc[i][j], C.ld, wmma::mem_row_major);
            }
        }
    }
}

void launch_sgemm_wmma(
    TensorView2D<half> A,
    TensorView2D<half> B,
    TensorView2D<float> C,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    // 128 threads = 4 warps
    dim3 block(128, 1, 1);
    
    // Grid handles the 64x64 blocks
    dim3 grid(
        (C.cols + 63) / 64,
        (C.rows + 63) / 64,
        1
    );

    sgemm_wmma_tiled_kernel<<<grid, block, 0, stream>>>(A, B, C, alpha, beta);
}

// 2. STANDARD SHARED MEMORY KERNEL
__global__ void sgemm_shared_mem_kernel(
    TensorView2D<float> A,
    TensorView2D<float> B,
    TensorView2D<float> C,
    float alpha,
    float beta
) {
    int M = A.rows;
    int N = B.cols;
    int K = A.cols;

    __shared__ float As[BLOCKSIZE][BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE][BLOCKSIZE];

    int globalRow = blockIdx.y * BLOCKSIZE + threadIdx.y;
    int globalCol = blockIdx.x * BLOCKSIZE + threadIdx.x;

    float tmp = 0.0f;

    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        int aRow = globalRow;
        int aCol = bkIdx + threadIdx.x;
        As[threadIdx.y][threadIdx.x] =
            (aRow < M && aCol < K) ? A.data[aRow * A.ld + aCol] : 0.0f;

        int bRow = bkIdx + threadIdx.y;
        int bCol = globalCol;
        Bs[threadIdx.y][threadIdx.x] =
            (bRow < K && bCol < N) ? B.data[bRow * B.ld + bCol] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BLOCKSIZE; ++k) {
            tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (globalRow < M && globalCol < N) {
        int cIdx = globalRow * C.ld + globalCol;
        C.data[cIdx] = alpha * tmp + beta * C.data[cIdx];
    }
}

void launch_gemm(
    TensorView2D<float> A,
    TensorView2D<float> B,
    TensorView2D<float> C,
    float alpha,
    float beta,
    cudaStream_t stream
) {
    dim3 blockDim(BLOCKSIZE, BLOCKSIZE);
    dim3 gridDim(
        (C.cols + BLOCKSIZE - 1) / BLOCKSIZE,
        (C.rows + BLOCKSIZE - 1) / BLOCKSIZE
    );

    sgemm_shared_mem_kernel<<<gridDim, blockDim, 0, stream>>>(A, B, C, alpha, beta);
}