#include <stdio.h>
#include <assert.h>
#include <mkl.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <cuComplex.h>

#include "Lap_Kron_GPU.h"
#include "tools.h"

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_Lap_Kron(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
                 double *d_vec, double *d_diag, double *d_out)
{
    cublasHandle_t handle;
    cublasStatus_t cubSt;
    cudaError_t cuE;

    cubSt = cublasCreate(&handle); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *d_vecTVy, *d_VxtvecTVy, *d_P, *d_PTVyt, *d_VxPTVyt;
    cuE = cudaMalloc((void **) &d_vecTVy, sizeof(double) * NxNy); assert(cudaSuccess == cuE);
    d_VxtvecTVy = d_out;
    cuE = cudaMalloc((void **) &d_P, sizeof(double) * Nd); assert(cudaSuccess == cuE);
    d_PTVyt = d_vecTVy;
    d_VxPTVyt = d_P;

    double alpha = 1.0, beta = 0;
    for (int k = 0; k < Nz; k++) {
        cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Ny, 
                    &alpha, d_vec + k*NxNy, Nx, d_Vy, Ny, &beta, d_vecTVy, Nx); 
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
        
        cubSt = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nx, Ny, Nx, 
                    &alpha, d_Vx, Nx, d_vecTVy, Nx, &beta, d_VxtvecTVy + k*NxNy, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    }

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NxNy, Nz, Nz, 
                    &alpha, d_VxtvecTVy, NxNy, d_Vz, Nz, &beta, d_P, NxNy);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    int numThreadsPerBlock = 256;
    int numBlocks = (Nd + numThreadsPerBlock - 1) / numThreadsPerBlock;

    Hammond_RR<<<numBlocks, numThreadsPerBlock>>>(d_P, d_diag, d_P, Nd);

    for (int k = 0; k < Nz; k++) {
        cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nx, Ny, Ny, 
                    &alpha, d_P + k*NxNy, Nx, d_Vy, Ny, &beta, d_PTVyt, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);

        cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Nx, 
                    &alpha, d_Vx, Nx, d_PTVyt, Nx, &beta, d_VxPTVyt + k*NxNy, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    }
    
    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NxNy, Nz, Nz, 
                    &alpha, d_VxPTVyt, NxNy, d_Vz, Nz, &beta, d_out, NxNy);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);
    
    cuE = cudaFree(d_vecTVy); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_P); assert(cudaSuccess == cuE);
    cubSt = cublasDestroy(handle); assert(CUBLAS_STATUS_SUCCESS == cubSt);
}


void CUDA_Lap_Kron_complex(int Nx, int Ny, int Nz, cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, 
                 cuDoubleComplex *d_VyH, cuDoubleComplex *d_VzH, cuDoubleComplex *d_vec, double *d_diag, cuDoubleComplex *d_out)
{
    cublasHandle_t handle;
    cublasStatus_t cubSt;
    cudaError_t cuE;

    cubSt = cublasCreate(&handle); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    
    cuDoubleComplex *d_vecTVy, *d_VxtvecTVy, *d_P, *d_PTVyt, *d_VxPTVyt;
    cuE = cudaMalloc((void **) &d_vecTVy, sizeof(cuDoubleComplex) * NxNy); assert(cudaSuccess == cuE);
    d_VxtvecTVy = d_out;    
    cuE = cudaMalloc((void **) &d_P, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);    
    d_PTVyt = d_vecTVy;    
    d_VxPTVyt = d_P;    
    cuDoubleComplex aplha = make_cuDoubleComplex(1.0, 0.0), beta = make_cuDoubleComplex(0.0, 0.0);

    // P = Lambda .* (VzH x VyH x VxH) * vec
    for (int k = 0; k < Nz; k++) {    
        cubSt = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Ny, 
                    &aplha, d_vec + k*NxNy, Nx, d_VyH, Ny, &beta, d_vecTVy, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    
        cubSt = cublasZgemm(handle, CUBLAS_OP_C, CUBLAS_OP_N, Nx, Ny, Nx, 
                    &aplha, d_Vx, Nx, d_vecTVy, Nx, &beta, d_VxtvecTVy + k*NxNy, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    }

    cubSt = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NxNy, Nz, Nz, 
                    &aplha, d_VxtvecTVy, NxNy, d_VzH, Nz, &beta, d_P, NxNy);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    int numThreadsPerBlock = 256;
    int numBlocks = (Nd + numThreadsPerBlock - 1) / numThreadsPerBlock;

    Hammond_CR<<<numBlocks, numThreadsPerBlock>>>(d_P, d_diag, d_P, Nd);

    // out = (Vz x Vy x Vx) * P
    for (int k = 0; k < Nz; k++) {
        cubSt = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nx, Ny, Ny, 
                    &aplha, d_P + k*NxNy, Nx, d_Vy, Ny, &beta, d_PTVyt, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    
        cubSt = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny, Nx, 
                    &aplha, d_Vx, Nx, d_PTVyt, Nx, &beta, d_VxPTVyt + k*NxNy, Nx);
        assert(CUBLAS_STATUS_SUCCESS == cubSt);
    }

    cubSt = cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NxNy, Nz, Nz, 
                    &aplha, d_VxPTVyt, NxNy, d_Vz, Nz, &beta, d_out, NxNy);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    cuE = cudaFree(d_vecTVy); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_P); assert(cudaSuccess == cuE);
    cubSt = cublasDestroy(handle); assert(CUBLAS_STATUS_SUCCESS == cubSt);
}


void CUDA_Lap_Kron_multicol(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
                 double *d_vec, int ncol, double *d_diag, double *d_out)
{
    cublasHandle_t handle;
    cublasStatus_t cubSt;
    cudaError_t cuE;

    cubSt = cublasCreate(&handle); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    int len = Nd * ncol;
    double *d_temp;
    cuE = cudaMalloc((void **) &d_temp, sizeof(double) * len); assert(cudaSuccess == cuE);

    double *d_VxTvec, *d_VxTvecVy, *d_P, *d_DP, *d_VxDP, *d_VxDpVyT, *d_res;
    double *d_FormVxTvec, *d_FormVxTvecVy, *d_FormDP, *d_FormVxDP, *d_FormVxDpVyT;
    
    d_VxTvec = d_VxTvecVy = d_P = d_FormDP = d_FormVxDP = d_FormVxDpVyT = d_out;
    d_FormVxTvec = d_FormVxTvecVy = d_DP = d_VxDP = d_VxDpVyT = d_res = d_temp;
    double alpha = 1.0, beta = 0;

    // P = Lambda .* (Vz' x Vy' x Vx') * vec
    cubSt = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, Nx, Ny*Nz*ncol, Nx, 
                &alpha, d_Vx, Nx, d_vec, Nx, &beta, d_VxTvec, Nx);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // f1Tof2(Nx, Ny, Nz, ncol, VxTvec, FormVxTvec, sizeof(double));
    f1Tof2_GPU(Nx, Ny, Nz, ncol, d_VxTvec, d_FormVxTvec, sizeof(double));

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx*Nz*ncol, Ny, Ny, 
                &alpha, d_FormVxTvec, Nx*Nz*ncol, d_Vy, Ny, &beta, d_VxTvecVy, Nx*Nz*ncol);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    f2Tof4_GPU(Nx, Ny, Nz, ncol, d_VxTvecVy, d_FormVxTvecVy, sizeof(double));

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, NxNy*ncol, Nz, Nz, 
                &alpha, d_FormVxTvecVy, NxNy*ncol, d_Vz, Nz, &beta, d_P, NxNy*ncol);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // // apply diagonal term
    apply_diagonal_GPU(Nx, Ny, Nz, ncol, d_P, d_diag, d_DP, sizeof(double));

    // // out = (Vz x Vy x Vx) * P
    f4Tof1_GPU(Nx, Ny, Nz, ncol, d_DP, d_FormDP, sizeof(double));

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, Nx, Ny*Nz*ncol, Nx, 
                &alpha, d_Vx, Nx, d_FormDP, Nx, &beta, d_VxDP, Nx);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    f1Tof2_GPU(Nx, Ny, Nz, ncol, d_VxDP, d_FormVxDP, sizeof(double));

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, Nx*Nz*ncol, Ny, Ny, 
                &alpha, d_FormVxDP, Nx*Nz*ncol, d_Vy, Ny, &beta, d_VxDpVyT, Nx*Nz*ncol);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    f2Tof4_GPU(Nx, Ny, Nz, ncol, d_VxDpVyT, d_FormVxDpVyT, sizeof(double));

    cubSt = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, NxNy*ncol, Nz, Nz, 
                &alpha, d_FormVxDpVyT, NxNy*ncol, d_Vz, Nz, &beta, d_res, NxNy*ncol);
    assert(CUBLAS_STATUS_SUCCESS == cubSt);

    f4Tof1_GPU(Nx, Ny, Nz, ncol, d_res, d_out, sizeof(double));

    cuE = cudaFree(d_temp); assert(cudaSuccess == cuE);
}




void f1Tof2_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{
    dim3 blockDims(8,8,8);
    dim3 gridDims((Nx + blockDims.x - 1) / blockDims.x, 
                  (Ny + blockDims.y - 1) / blockDims.y, 
                  (Nz * ncol + blockDims.z - 1) / blockDims.z);

    
    f1Tof2_kernel<<<gridDims, blockDims>>>(Nx, Ny, Nz, ncol, d_src, d_dest, unit_size);
}

__global__ void f1Tof2_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{    
    int NxNy = Nx * Ny;
    int Nd = NxNy * Nz;
    int NxNzncol = Nx * Nz * ncol;
    // int len = NxNzncol * Ny;

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int combinedIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int idx_z = combinedIdx % Nz;
    int idx_c = combinedIdx / Nz;

    if (idx_x >= Nx || idx_y >= Ny || idx_z >= Nz || idx_c >= ncol) return;

    // Determine index for src
    int idx_src = idx_x + idx_y * Nx + idx_z * NxNy + idx_c * Nd;
    // Determine index for dst
    int idx_dst = idx_x + combinedIdx * Nx + idx_y * NxNzncol;
    
    if (unit_size == sizeof(double)) {
        double *d_src_ = (double *) d_src;
        double *d_dest_ = (double *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }

    if (unit_size == sizeof(cuDoubleComplex)) {
        cuDoubleComplex *d_src_ = (cuDoubleComplex *) d_src;
        cuDoubleComplex *d_dest_ = (cuDoubleComplex *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }
}


void f2Tof4_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{
    dim3 blockDims(8,8,8);
    dim3 gridDims((Nx + blockDims.x - 1) / blockDims.x, 
                  (Ny + blockDims.y - 1) / blockDims.y, 
                  (Nz * ncol + blockDims.z - 1) / blockDims.z);

    f2Tof4_kernel<<<gridDims, blockDims>>>(Nx, Ny, Nz, ncol, d_src, d_dest, unit_size);
}


__global__ void f2Tof4_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{
    int NxNy = Nx * Ny;
    int NxNzncol = Nx * Nz * ncol;
    int NxNyncol = NxNy * ncol;    

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int combinedIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int idx_z = combinedIdx % Nz;
    int idx_c = combinedIdx / Nz;

    if (idx_x >= Nx || idx_y >= Ny || idx_z >= Nz || idx_c >= ncol) return;

    // Determine index for src
    int idx_src = idx_x + combinedIdx * Nx + idx_y * NxNzncol;
    // Determine index for dst
    int idx_dst = idx_x + idx_y * Nx + idx_z * NxNyncol + idx_c * NxNy;

    if (unit_size == sizeof(double)) {
        double *d_src_ = (double *) d_src;
        double *d_dest_ = (double *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }

    if (unit_size == sizeof(cuDoubleComplex)) {
        cuDoubleComplex *d_src_ = (cuDoubleComplex *) d_src;
        cuDoubleComplex *d_dest_ = (cuDoubleComplex *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }
}


void f4Tof1_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{
    dim3 blockDims(8,8,8);
    dim3 gridDims((Nx + blockDims.x - 1) / blockDims.x, 
                  (Ny + blockDims.y - 1) / blockDims.y, 
                  (Nz * ncol + blockDims.z - 1) / blockDims.z);

    f4Tof1_kernel<<<gridDims, blockDims>>>(Nx, Ny, Nz, ncol, d_src, d_dest, unit_size);
}


__global__ void f4Tof1_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size) 
{
    int NxNy = Nx * Ny;
    int Nd = NxNy * Nz;
    int NxNyncol = NxNy * ncol;

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int combinedIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int idx_z = combinedIdx % Nz;
    int idx_c = combinedIdx / Nz;

    if (idx_x >= Nx || idx_y >= Ny || idx_z >= Nz || idx_c >= ncol) return;

    // Determine index for src
    int idx_src = idx_x + idx_y * Nx + idx_z * NxNyncol + idx_c * NxNy;
    // Determine index for dst
    int idx_dst = idx_x + idx_y * Nx + idx_z * NxNy + idx_c * Nd;

    if (unit_size == sizeof(double)) {
        double *d_src_ = (double *) d_src;
        double *d_dest_ = (double *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }

    if (unit_size == sizeof(cuDoubleComplex)) {
        cuDoubleComplex *d_src_ = (cuDoubleComplex *) d_src;
        cuDoubleComplex *d_dest_ = (cuDoubleComplex *) d_dest;
        d_dest_[idx_dst] = d_src_[idx_src];
    }
}

void apply_diagonal_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_in, double *d_diag, void *d_out, const size_t unit_size)
{
    dim3 blockDims(8,8,8);
    dim3 gridDims((Nx + blockDims.x - 1) / blockDims.x, 
                  (Ny + blockDims.y - 1) / blockDims.y, 
                  (Nz * ncol + blockDims.z - 1) / blockDims.z);

    apply_diagonal_kernel<<<gridDims, blockDims>>>(Nx, Ny, Nz, ncol, d_in, d_diag, d_out, unit_size);
}


__global__ void apply_diagonal_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_in, double *d_diag, void *d_out, const size_t unit_size)
{
    int NxNy = Nx * Ny;
    int NxNyncol = NxNy * ncol;    

    int idx_x = blockIdx.x * blockDim.x + threadIdx.x;
    int idx_y = blockIdx.y * blockDim.y + threadIdx.y;
    int combinedIdx = blockIdx.z * blockDim.z + threadIdx.z;
    int idx_z = combinedIdx % Nz;
    int idx_c = combinedIdx / Nz;

    if (idx_x >= Nx || idx_y >= Ny || idx_z >= Nz || idx_c >= ncol) return;

    // Determine index for src
    int idx_vec = idx_x + idx_y * Nx + idx_z * NxNyncol + idx_c * NxNy;
    // Determine index for dst
    int idx_diag = idx_x + idx_y * Nx + idx_z * NxNy;

    if (unit_size == sizeof(double)) {
        double *d_in_ = (double *) d_in;
        double *d_out_ = (double *) d_out;
        double *d_diag_ = (double *) d_diag;
        d_out_[idx_vec] = d_in_[idx_vec] * d_diag_[idx_diag];
    }

    if (unit_size == sizeof(cuDoubleComplex)) {
        cuDoubleComplex *d_in_ = (cuDoubleComplex *) d_in;
        cuDoubleComplex *d_out_ = (cuDoubleComplex *) d_out;
        double *d_diag_ = (double *) d_diag;

        cuDoubleComplex res;
        res.x = d_in_[idx_vec].x * d_diag_[idx_diag];
		res.y = d_in_[idx_vec].y * d_diag_[idx_diag];
        d_out_[idx_vec] = res;
    }
}



#ifdef __cplusplus
}
#endif