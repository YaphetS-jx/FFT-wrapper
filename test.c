#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/time.h>    // for gettimeofday()

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>

#include "test.h"
#include "fft.h"
#include "tools.h"
#include "Lap_Matrix.h"
#include "Lap_Kron.h"
#include "Lap_Kron_GPU.h"
#include "fft_GPU.h"

void Kron_compare_single_col_GPU(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *eig,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_eig, int reps)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
    double *LapX = (double *) malloc(sizeof(double) * Nd);
    double *LapX_gpu = (double *) malloc(sizeof(double) * Nd);
    rand_vec(X, Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    double *d_X, *d_LapX;
    cuE = cudaMalloc((void **) &d_X, sizeof(double) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_LapX, sizeof(double) * Nd); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(double), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, eig, LapX);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_Kron_CUDA(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_X, d_eig, d_LapX);
    }
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    cubSt = cublasGetVector(Nd, sizeof(double), d_LapX, 1, LapX_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    printf("==============================================\n");
    printf("err between CPU and GPU for single column way: %.2e\n", err(LapX,LapX_gpu,Nd));
    printf("%d times cup takes  : %.1f ms\n", reps, t_cpu);
    printf("%d times gpu takes  : %.1f ms\n", reps, t_gpu);  
    printf("t_cpu/t_gpu -1: %.1f %%\n", (t_cpu/t_gpu -1)*100);
    printf("==============================================\n");

    free(X);
    free(LapX);
    free(LapX_gpu);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
}


void compare_FFT_CPU_GPU(int Nx, int Ny, int Nz, int reps)
{
    int Nd = Nx * Ny * Nz;
    int Nd_half = (Nx/2+1)*Ny*Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
    double _Complex *Xbar = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);
    double _Complex *Xbar_gpu = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);
    double *X2 = (double *) malloc(sizeof(double) * Nd);
    rand_vec(X, Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cufftDoubleReal *d_X, *d_X2; cuDoubleComplex *d_Xbar;
    cuE = cudaMalloc((void **) &d_X, sizeof(cufftDoubleReal) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_X2, sizeof(cufftDoubleReal) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_Xbar, sizeof(cuDoubleComplex) * Nd_half); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(cufftDoubleReal), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // timing variable 
    struct timeval start, end;

    // CPU fft
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        MKL_MDFFT_real(X, dim_sizes, strides_out, Xbar);
        MKL_MDiFFT_real(Xbar, dim_sizes, strides_out, X2);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;    
    assert(err(X,X2,Nd) < 1E-8);

    // GPU fft
    int dim_sizes_gpu[3] = {Nz, Ny, Nx};
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        CUDA_MDFFT_real(d_X, dim_sizes_gpu, d_Xbar);        
        CUDA_MDiFFT_real(d_Xbar, dim_sizes_gpu, d_X2);
    }
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    cubSt = cublasGetVector(Nd, sizeof(cufftDoubleReal), d_X2, 1, X2, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    assert(err(X,X2,Nd) < 1E-8);

    cubSt = cublasGetVector(Nd_half, sizeof(cuDoubleComplex), d_Xbar, 1, Xbar_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    printf("==============================================\n");
    printf("err between CPU and GPU FFT: %.2e\n", err((double *)Xbar, (double *)Xbar_gpu, 2*Nd_half));
    printf("%d times cup takes  : %.1f ms\n", reps, t_cpu);
    printf("%d times gpu takes  : %.1f ms\n", reps, t_gpu);  
    printf("t_cpu/t_gpu -1: %.1f %%\n", (t_cpu/t_gpu -1)*100);
    printf("==============================================\n");
    
    free(X);
    free(Xbar);
    free(Xbar_gpu);
    free(X2);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_X2); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_Xbar); assert(cudaSuccess == cuE);
}