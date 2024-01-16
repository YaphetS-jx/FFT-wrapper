#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>
#include <mkl.h>

#include "Lap_Kron_GPU.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void elementWiseMultiply(double* a, double* b, double* c, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}


void Lap_Kron_CUDA(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
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

	elementWiseMultiply<<<numBlocks, numThreadsPerBlock>>>(d_P, d_diag, d_P, Nd);

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



#ifdef __cplusplus
}
#endif