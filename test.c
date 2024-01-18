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




////////////////////////////////////////////////////////
////////                FFT tests               ////////
////////////////////////////////////////////////////////

void verify_FFT_CPU_GPU(int Nx, int Ny, int Nz)
{
    int Nd = Nx * Ny * Nz;
    int Nd_half = (Nx/2+1)*Ny*Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
    double *X_GPU = (double *) malloc(sizeof(double) * Nd);
    double _Complex *Xbar = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);
    double _Complex *Xbar_gpu = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);
    
    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cufftDoubleReal *d_X; cuDoubleComplex *d_Xbar;
    cuE = cudaMalloc((void **) &d_X, sizeof(cufftDoubleReal) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_Xbar, sizeof(cuDoubleComplex) * Nd_half); assert(cudaSuccess == cuE);

    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 
    int dim_sizes_gpu[3] = {Nz, Ny, Nx};    

    // forward
    rand_vec(X, Nd);
    cubSt = cublasSetVector(Nd, sizeof(cufftDoubleReal), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    MKL_MDFFT_real(X, dim_sizes, strides_out, Xbar);
    
    CUDA_MDFFT_real(d_X, dim_sizes_gpu, d_Xbar);  

    cubSt = cublasGetVector(Nd_half, sizeof(cuDoubleComplex), d_Xbar, 1, Xbar_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    printf("err between CPU and GPU FFT: %.2e\n", err((double *)Xbar, (double *)Xbar_gpu, 2*Nd_half));

    // backward
    MKL_MDiFFT_real(Xbar, dim_sizes, strides_out, X);

    CUDA_MDiFFT_real(d_Xbar, dim_sizes_gpu, d_X);

    cubSt = cublasGetVector(Nd, sizeof(cufftDoubleReal), d_X, 1, X_GPU, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    printf("err between CPU and GPU iFFT: %.2e\n", err(X, X_GPU, Nd));    
    
    free(X);
    free(X_GPU);
    free(Xbar);
    free(Xbar_gpu);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_Xbar); assert(cudaSuccess == cuE);
}


void verify_FFT_CPU_GPU_complex(int Nx, int Ny, int Nz)
{
    int Nd = Nx * Ny * Nz;    
    double _Complex *X = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *X_GPU = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *Xbar = (double _Complex *) malloc(sizeof(double _Complex) * Nd);
    double _Complex *Xbar_gpu = (double _Complex *) malloc(sizeof(double _Complex) * Nd);
    
    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cuDoubleComplex *d_X, *d_Xbar;
    cuE = cudaMalloc((void **) &d_X, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_Xbar, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);

    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*Nx, Nx, 1}; 
    int dim_sizes_gpu[3] = {Nz, Ny, Nx};    

    // forward
    rand_vec((double *)X, 2*Nd);
    cubSt = cublasSetVector(Nd, sizeof(cuDoubleComplex), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    MKL_MDFFT(X, dim_sizes, strides_out, Xbar);
    
    CUDA_MDFFT(d_X, dim_sizes_gpu, d_Xbar);

    cubSt = cublasGetVector(Nd, sizeof(cuDoubleComplex), d_Xbar, 1, Xbar_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    printf("err between CPU and GPU FFT complex: %.2e\n", err((double *)Xbar, (double *)Xbar_gpu, 2*Nd));

    // backward

    MKL_MDiFFT(Xbar, dim_sizes, strides_out, X);

    CUDA_MDiFFT(d_Xbar, dim_sizes_gpu, d_X);

    cubSt = cublasGetVector(Nd, sizeof(cuDoubleComplex), d_X, 1, X_GPU, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    printf("err between CPU and GPU iFFT complex: %.2e\n", err((double *)X, (double *)X_GPU, 2*Nd));    
    
    free(X);
    free(X_GPU);
    free(Xbar);
    free(Xbar_gpu);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_Xbar); assert(cudaSuccess == cuE);
}


void FFT_iFFT_CPU(int Nx, int Ny, int Nz, int reps)
{
    int Nd = Nx * Ny * Nz;
    int Nd_half = (Nx/2+1)*Ny*Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
    double _Complex *Xbar = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);
    rand_vec(X, Nd);

    // timing variable 
    struct timeval start, end;

    // CPU fft
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        MKL_MDFFT_real(X, dim_sizes, strides_out, Xbar);
        MKL_MDiFFT_real(Xbar, dim_sizes, strides_out, X);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3; 
    
    printf("FFT_iFFT_CPU %d times takes on average : %.1f ms\n", reps, t_cpu/reps);
    
    free(X);
    free(Xbar);    
}

void FFT_iFFT_complex_CPU(int Nx, int Ny, int Nz, int reps)
{
    int Nd = Nx * Ny * Nz;    
    double _Complex *X = (double _Complex *) malloc(sizeof(double _Complex) * Nd);
    double _Complex *Xbar = (double _Complex *) malloc(sizeof(double _Complex) * Nd);
    rand_vec((double *)X, 2*Nd);

    // timing variable 
    struct timeval start, end;

    // CPU fft
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*Nx, Nx, 1}; 

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        MKL_MDFFT(X, dim_sizes, strides_out, Xbar);
        MKL_MDiFFT(Xbar, dim_sizes, strides_out, X);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3; 
    
    printf("FFT_iFFT_complex_CPU %d times takes on average : %.1f ms\n", reps, t_cpu/reps);
    
    free(X);
    free(Xbar);    
}


void FFT_iFFT_GPU(int Nx, int Ny, int Nz, int reps)
{
    int Nd = Nx * Ny * Nz;
    int Nd_half = (Nx/2+1)*Ny*Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);    
    rand_vec(X, Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cufftDoubleReal *d_X; cuDoubleComplex *d_Xbar;
    cuE = cudaMalloc((void **) &d_X, sizeof(cufftDoubleReal) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_Xbar, sizeof(cuDoubleComplex) * Nd_half); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(cufftDoubleReal), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // timing variable 
    struct timeval start, end;

    // cudaEvent_t custart, custop;
    // cudaEventCreate(&custart); cudaEventCreate(&custop);

    // GPU fft
    int dim_sizes_gpu[3] = {Nz, Ny, Nx};
    gettimeofday( &start, NULL );

    // cudaEventRecord(custart,0);

    for (int rep = 0; rep < reps; rep++) {
        CUDA_MDFFT_real(d_X, dim_sizes_gpu, d_Xbar);        
        CUDA_MDiFFT_real(d_Xbar, dim_sizes_gpu, d_X);
    }

    // cudaEventRecord(custop,0);
    // cudaEventSynchronize(custop);

    // float t_gpu_cu = 0;
    // cudaEventElapsedTime(&t_gpu_cu, custart, custop);

    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("FFT_iFFT_GPU %d times takes on average : %.1f ms\n", reps, t_gpu/reps);
    // printf("FFT_iFFT_GPU %d times takes on average (CUDA count): %.1f ms\n", reps, t_gpu_cu/reps);
    
    free(X);    
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_Xbar); assert(cudaSuccess == cuE);
}


void FFT_iFFT_complex_GPU(int Nx, int Ny, int Nz, int reps)
{
    int Nd = Nx * Ny * Nz;    
    double _Complex *X = (double _Complex*) malloc(sizeof(double _Complex) * Nd);    
    rand_vec((double *)X, Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cuDoubleComplex *d_X, *d_Xbar;
    cuE = cudaMalloc((void **) &d_X, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_Xbar, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(cuDoubleComplex), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // timing variable 
    struct timeval start, end;

    // GPU fft
    int dim_sizes_gpu[3] = {Nz, Ny, Nx};
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        CUDA_MDFFT(d_X, dim_sizes_gpu, d_Xbar);        
        CUDA_MDiFFT(d_Xbar, dim_sizes_gpu, d_X);
    }
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("FFT_iFFT_complex_GPU %d times takes on average : %.1f ms\n", reps, t_gpu/reps);
    
    free(X);    
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_Xbar); assert(cudaSuccess == cuE);
}


////////////////////////////////////////////////////////
////////         Kron single col tests          ////////
////////////////////////////////////////////////////////

void verify_single_col(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *diag,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag)
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
    
    Lap_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, diag, LapX);    

    CUDA_Lap_Kron(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_X, d_diag, d_LapX);

    cubSt = cublasGetVector(Nd, sizeof(double), d_LapX, 1, LapX_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
        
    printf("err between CPU and GPU for single column way: %.2e\n", err(LapX,LapX_gpu,Nd));    

    free(X);
    free(LapX);
    free(LapX_gpu);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
}


void kron_single_col_CPU(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
    double *LapX = (double *) malloc(sizeof(double) * Nd);
    rand_vec(X, Nd);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, diag, LapX);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_single_col_CPU %d times takes on average : %.2f ms\n", reps, t_cpu/reps);

    free(X);
    free(LapX);    
}


void kron_single_col_GPU(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd);
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
        CUDA_Lap_Kron(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_X, d_diag, d_LapX);
    }
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_single_col_GPU %d times takes on average : %.2f ms\n", reps, t_gpu/reps);

    free(X);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
}



void verify_single_col_complex(int Nx, int Ny, int Nz, 
                double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag,
                cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag)
{
    int Nd = Nx * Ny * Nz;
    double _Complex *X = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *LapX = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *LapX_gpu = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *VyH = (double _Complex*) malloc(sizeof(double _Complex) * Ny*Ny);
    double _Complex *VzH = (double _Complex*) malloc(sizeof(double _Complex) * Nz*Nz);
    for (int i = 0; i < Ny*Ny; i++) VyH[i] = conj(Vy[i]);
    for (int i = 0; i < Nz*Nz; i++) VzH[i] = conj(Vz[i]);
    rand_vec((double *)X, 2*Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cuDoubleComplex *d_X, *d_LapX, *d_VyH, *d_VzH;
    cuE = cudaMalloc((void **) &d_X, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_LapX, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_VyH, sizeof(cuDoubleComplex) * Ny*Ny); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_VzH, sizeof(cuDoubleComplex) * Nz*Nz); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(cuDoubleComplex), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);    
    cubSt = cublasSetVector(Ny*Ny, sizeof(cuDoubleComplex), VyH, 1, d_VyH, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);    
    cubSt = cublasSetVector(Nz*Nz, sizeof(cuDoubleComplex), VzH, 1, d_VzH, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);    
    
    Lap_Kron_complex(Nx, Ny, Nz, Vx, Vy, Vz, VyH, VzH, X, diag, LapX);

    CUDA_Lap_Kron_complex(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_VyH, d_VzH, d_X, d_diag, d_LapX);

    cubSt = cublasGetVector(Nd, sizeof(cuDoubleComplex), d_LapX, 1, LapX_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    
    printf("err between CPU and GPU for single column way complex: %.2e\n", err((double *)LapX, (double *)LapX_gpu, 2*Nd));    

    free(X);
    free(LapX);
    free(LapX_gpu);
    free(VyH);
    free(VzH);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_VyH); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_VzH); assert(cudaSuccess == cuE);
}


void kron_single_col_complex_CPU(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double _Complex *X = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *LapX = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *LapX_gpu = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *VyH = (double _Complex*) malloc(sizeof(double _Complex) * Ny*Ny);
    double _Complex *VzH = (double _Complex*) malloc(sizeof(double _Complex) * Nz*Nz);
    for (int i = 0; i < Ny*Ny; i++) VyH[i] = conj(Vy[i]);
    for (int i = 0; i < Nz*Nz; i++) VzH[i] = conj(Vz[i]);
    rand_vec((double *)X, 2*Nd);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_Kron_complex(Nx, Ny, Nz, Vx, Vy, Vz, VyH, VzH, X, diag, LapX);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_single_col_complex_CPU %d times takes on average : %.2f ms\n", reps, t_cpu/reps);
    

    free(X);
    free(LapX);
    free(LapX_gpu);
    free(VyH);
    free(VzH);
}


void kron_single_col_complex_GPU(int Nx, int Ny, int Nz, cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double _Complex *X = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    rand_vec((double *)X, 2*Nd);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    cuDoubleComplex *d_X, *d_LapX, *d_VyH, *d_VzH;
    cuE = cudaMalloc((void **) &d_X, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_LapX, sizeof(cuDoubleComplex) * Nd); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_VyH, sizeof(cuDoubleComplex) * Ny*Ny); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_VzH, sizeof(cuDoubleComplex) * Nz*Nz); assert(cudaSuccess == cuE);
    cuE = cudaMemcpy(d_VyH, d_Vy, sizeof(cuDoubleComplex) * Ny*Ny, cudaMemcpyDeviceToDevice); assert(cudaSuccess == cuE);
    cuE = cudaMemcpy(d_VzH, d_Vz, sizeof(cuDoubleComplex) * Nz*Nz, cudaMemcpyDeviceToDevice); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd, sizeof(cuDoubleComplex), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    
    conjugate_vector(d_VyH, Ny*Ny);
    conjugate_vector(d_VzH, Nz*Nz);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        CUDA_Lap_Kron_complex(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_VyH, d_VzH, d_X, d_diag, d_LapX);
    }
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_single_col_complex_GPU %d times takes on average : %.2f ms\n", reps, t_gpu/reps);

    free(X);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_VyH); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_VzH); assert(cudaSuccess == cuE);
}



////////////////////////////////////////////////////////
////////         Kron multiple col tests        ////////
////////////////////////////////////////////////////////


void verify_multiple_col(int Nx, int Ny, int Nz, int ncol, double *Vx, double *Vy, double *Vz, double *diag,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd*ncol);
    double *LapX = (double *) malloc(sizeof(double) * Nd*ncol);
    double *LapX_gpu = (double *) malloc(sizeof(double) * Nd*ncol);
    rand_vec(X, Nd*ncol);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    double *d_X, *d_LapX;
    cuE = cudaMalloc((void **) &d_X, sizeof(double) * Nd*ncol); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_LapX, sizeof(double) * Nd*ncol); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd*ncol, sizeof(double), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);    
    
    Lap_Kron_multicol(Nx, Ny, Nz, Vx, Vy, Vz, X, ncol, diag, LapX);
    
    CUDA_Lap_Kron_multicol(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_X, ncol, d_diag, d_LapX);

    cubSt = cublasGetVector(Nd*ncol, sizeof(double), d_LapX, 1, LapX_gpu, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    printf("err between CPU and GPU for multiple column way: %.2e\n", err(LapX,LapX_gpu,Nd*ncol));    

    free(X);
    free(LapX);
    free(LapX_gpu);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
}


void kron_multiple_col_CPU(int Nx, int Ny, int Nz, int ncol, double *Vx, double *Vy, double *Vz, double *diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd*ncol);
    double *LapX = (double *) malloc(sizeof(double) * Nd*ncol);
    rand_vec(X, Nd*ncol);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_Kron_multicol(Nx, Ny, Nz, Vx, Vy, Vz, X, ncol, diag, LapX);
    }
    gettimeofday( &end, NULL );
    double t_cpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_multiple_col_CPU %d times takes on average : %.2f ms\n", reps, t_cpu/reps);

    free(X);
    free(LapX);    
}


void kron_multiple_col_GPU(int Nx, int Ny, int Nz, int ncol, double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag, int reps)
{
    int Nd = Nx * Ny * Nz;
    double *X = (double *) malloc(sizeof(double) * Nd*ncol);
    rand_vec(X, Nd*ncol);

    // gpu variables
    cudaError_t cuE; cublasStatus_t cubSt;
    double *d_X, *d_LapX;
    cuE = cudaMalloc((void **) &d_X, sizeof(double) * Nd*ncol); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_LapX, sizeof(double) * Nd*ncol); assert(cudaSuccess == cuE);
    cubSt = cublasSetVector(Nd*ncol, sizeof(double), X, 1, d_X, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);

    // timing variable 
    struct timeval start, end;
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        CUDA_Lap_Kron_multicol(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_X, ncol, d_diag, d_LapX);
    }
    cudaDeviceSynchronize();
    gettimeofday( &end, NULL );
    double t_gpu = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    printf("Kron_multiple_col_GPU %d times takes on average : %.2f ms\n", reps, t_gpu/reps);

    free(X);
    cuE = cudaFree(d_X); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_LapX); assert(cudaSuccess == cuE);
}