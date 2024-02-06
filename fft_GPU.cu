#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>

#include "fft_GPU.h"
#include "tools.h"


#ifdef __cplusplus
extern "C" {
#endif



void CUDA_fft_solve(int Nx, int Ny, int Nz, double *d_rhs, double *d_pois_FFT_const, double *d_sol)
{
    int Nd_half = (Nx/2+1)*Ny*Nz;
    int dim_sizes[3] = {Nz, Ny, Nx};

    cudaError_t cuE;

    cuDoubleComplex *d_rhs_bar;
    cuE = cudaMalloc((void **) &d_rhs_bar, sizeof(cuDoubleComplex) * Nd_half); assert(cudaSuccess == cuE);

    CUDA_MDFFT_real(d_rhs, dim_sizes, d_rhs_bar);  

    int numThreadsPerBlock = 256;
	int numBlocks = (Nd_half + numThreadsPerBlock - 1) / numThreadsPerBlock;

	Hammond_CR<<<numBlocks, numThreadsPerBlock>>>(d_rhs_bar, d_pois_FFT_const, d_rhs_bar, Nd_half);

    CUDA_MDiFFT_real(d_rhs_bar, dim_sizes, d_sol);

    cuE = cudaFree(d_rhs_bar); assert(cudaSuccess == cuE);
}


/**
 * @brief   CUDA multi-dimension FFT interface, real to complex, following conjugate even distribution. 
 */
void CUDA_MDFFT_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_r2c_3doutput)
{
    cufftHandle plan_r2c;
    cufftCreate(&plan_r2c);
    cufftPlan3d(&plan_r2c, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_D2Z);
    cufftExecD2Z(plan_r2c, d_r2c_3dinput, d_r2c_3doutput);
    cufftDestroy(plan_r2c);
}

/**
 * @brief   CUDA multi-dimension FFT interface, complex to real, following conjugate even distribution. 
 *          Warning: d_c2r_3dinput will be changed!
 */
void CUDA_MDiFFT_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, cufftDoubleReal *d_c2r_3doutput)
{
    cufftHandle plan_c2r;
    cufftCreate(&plan_c2r);
    cufftPlan3d(&plan_c2r, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_Z2D);
    cufftExecZ2D(plan_c2r, d_c2r_3dinput, d_c2r_3doutput);
    cufftDestroy(plan_c2r);

    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    double scale = 1.0/N;
    scale_vector<<<numBlocks,numThreadsPerBlock>>>(d_c2r_3doutput, scale, N);
}


/**
 * @brief   CUDA multi-dimension FFT interface, complex to complex
 */
void CUDA_MDFFT(cufftDoubleComplex *d_c2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_c2c_3doutput)
{
    cufftHandle plan_c2c;
    cufftCreate(&plan_c2c);
    cufftPlan3d(&plan_c2c, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_Z2Z);
    cufftExecZ2Z(plan_c2c, d_c2c_3dinput, d_c2c_3doutput, CUFFT_FORWARD);
    cufftDestroy(plan_c2c);
}

/**
 * @brief   CUDA multi-dimension FFT interface, complex to complex
 */
void CUDA_MDiFFT(cufftDoubleComplex *d_c2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_c2c_3doutput)
{
    cufftHandle plan_c2c;
    cufftCreate(&plan_c2c);
    cufftPlan3d(&plan_c2c, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_Z2Z);
    cufftExecZ2Z(plan_c2c, d_c2c_3dinput, d_c2c_3doutput, CUFFT_INVERSE);
    cufftDestroy(plan_c2c);

    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    double scale = 1.0/N;
    scale_vector_complex<<<numBlocks,numThreadsPerBlock>>>(d_c2c_3doutput, scale, N);
}

// batched FFT
/**
 * @brief   CUDA multi-dimension batch FFT interface, real to complex, following conjugate even distribution. 
 */
void CUDA_MDFFT_batch_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, int ncol, cufftDoubleComplex *d_r2c_3doutput)
{
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];

    cufftHandle plan_r2c;
    cufftCreate(&plan_r2c);
    cufftPlanMany(&plan_r2c, 3, dim_sizes, NULL, 1, N, NULL, 1, N, CUFFT_D2Z, ncol);
    // cufftPlanMany(&plan_r2c, 3, dim_sizes, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, ncol);
    cufftExecD2Z(plan_r2c, d_r2c_3dinput, d_r2c_3doutput);
    cufftDestroy(plan_r2c);
}


/**
 * @brief   CUDA multi-dimension batch FFT interface, complex to real, following conjugate even distribution. 
 *          Warning: d_c2r_3dinput will be changed!
 */
void CUDA_MDiFFT_batch_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, int ncol, cufftDoubleReal *d_c2r_3doutput)
{
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];

    cufftResult_t res_t;

    cufftHandle plan_c2r;
    cufftCreate(&plan_c2r);
    // cufftPlanMany(&plan_c2r, 3, dim_sizes, NULL, 1, N, NULL, 1, N, CUFFT_Z2D, ncol);
    res_t = cufftPlanMany(&plan_c2r, 3, dim_sizes, NULL, 1, N, NULL, 1, N, CUFFT_Z2D, ncol);
    assert(res_t == CUFFT_SUCCESS);

    res_t = cufftExecZ2D(plan_c2r, d_c2r_3dinput, d_c2r_3doutput);
    assert(res_t == CUFFT_SUCCESS);
    
    cufftDestroy(plan_c2r);

    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    double scale = 1.0/N;
    scale_vector<<<numBlocks,numThreadsPerBlock>>>(d_c2r_3doutput, scale, N*ncol);
}


#ifdef __cplusplus
}
#endif