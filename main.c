#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>    // for gettimeofday()
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <assert.h>

#include "tools.h"
#include "Lap_Matrix.h"
#include "test.h"


int main(int argc, char *argv[]){

    int Nx, Ny, Nz, reps, ncol;
    Nx = Ny = Nz = 20;
    reps = 30;
    ncol = 10;
    if( argc == 4 ) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
    } else if (argc == 2) {
        reps = atoi(argv[1]);
    } else if (argc == 5) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
        reps = atoi(argv[4]);
    } else if (argc == 6) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
        reps = atoi(argv[4]);
        ncol = atoi(argv[5]);
    }
    printf("================\n");
    printf("Nx,Ny,Nz: %d,%d,%d\n", Nx, Ny, Nz);
    printf("reps: %d, ncol: %d\n", reps, ncol);

    srand(5);    

    verify_FFT_CPU_GPU(Nx, Ny, Nz);
    verify_FFT_CPU_GPU_complex(Nx, Ny, Nz);    
    FFT_iFFT_CPU(Nx, Ny, Nz, reps);
    FFT_iFFT_GPU(Nx, Ny, Nz, reps);    
    FFT_iFFT_complex_CPU(Nx, Ny, Nz, reps);
    FFT_iFFT_complex_GPU(Nx, Ny, Nz, reps);

        
    /******************************************************/
    /*********          Lap_1D decomposition        *******/
    /******************************************************/

    double dx = 0.3, dy = 0.5, dz = 0.2;
    double kptTL_x, kptTL_y, kptTL_z;
    kptTL_x = 1; kptTL_y = 1; kptTL_z = 1;
    double _Complex phase_fac_x = cos(kptTL_x) + I * sin(kptTL_x);
    double _Complex phase_fac_y = cos(kptTL_y) + I * sin(kptTL_y);
    double _Complex phase_fac_z = cos(kptTL_z) + I * sin(kptTL_z);
    int FDn = 6, Nd = Nx*Ny*Nz;

    double *FDweights_D2_x = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_y = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_z = (double *) malloc((FDn+1) * sizeof(double));
    double *Vx = (double *) malloc(Nx*Nx * sizeof(double));
    double *Vy = (double *) malloc(Ny*Ny * sizeof(double));
    double *Vz = (double *) malloc(Nz*Nz * sizeof(double));
    double _Complex *Vx_kpt = (double _Complex*) malloc(Nx*Nx * sizeof(double _Complex));
    double _Complex *Vy_kpt = (double _Complex*) malloc(Ny*Ny * sizeof(double _Complex));
    double _Complex *Vz_kpt = (double _Complex*) malloc(Nz*Nz * sizeof(double _Complex));
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    double *eig = (double *) calloc(sizeof(double),  Nd);

////////////////////////////////////////////
    cudaError_t cuE; cublasStatus_t cubSt;
    double *d_Vx, *d_Vy, *d_Vz, *d_eig;    
    cuE = cudaMalloc((void **) &d_Vx, sizeof(double) * Nx * Nx); assert(cudaSuccess == cuE);
	cuE = cudaMalloc((void **) &d_Vy, sizeof(double) * Ny * Ny); assert(cudaSuccess == cuE);
	cuE = cudaMalloc((void **) &d_Vz, sizeof(double) * Nz * Nz); assert(cudaSuccess == cuE);
    cuE = cudaMalloc((void **) &d_eig, sizeof(double) * Nx * Ny * Nz); assert(cudaSuccess == cuE);
    cuDoubleComplex *d_Vx_kpt, *d_Vy_kpt, *d_Vz_kpt;
    cuE = cudaMalloc((void **) &d_Vx_kpt, sizeof(cuDoubleComplex) * Nx * Nx); assert(cudaSuccess == cuE);
	cuE = cudaMalloc((void **) &d_Vy_kpt, sizeof(cuDoubleComplex) * Ny * Ny); assert(cudaSuccess == cuE);
	cuE = cudaMalloc((void **) &d_Vz_kpt, sizeof(cuDoubleComplex) * Nz * Nz); assert(cudaSuccess == cuE);

///////////////////////////////////////////

    calculate_FDweights_D2(FDn, dx, FDweights_D2_x);
    calculate_FDweights_D2(FDn, dy, FDweights_D2_y);
    calculate_FDweights_D2(FDn, dz, FDweights_D2_z);

//     // Dirichlet
//     Lap_1D_D_EigenDecomp(Nx, FDn, FDweights_D2_x, Vx, lambda_x);
//     Lap_1D_D_EigenDecomp(Ny, FDn, FDweights_D2_y, Vy, lambda_y);
//     Lap_1D_D_EigenDecomp(Nz, FDn, FDweights_D2_z, Vz, lambda_z);
//     eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eig);

    // Periodic real
    Lap_1D_P_EigenDecomp(Nx, FDn, FDweights_D2_x, Vx, lambda_x);
    Lap_1D_P_EigenDecomp(Ny, FDn, FDweights_D2_y, Vy, lambda_y);
    Lap_1D_P_EigenDecomp(Nz, FDn, FDweights_D2_z, Vz, lambda_z);
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eig);

////////////////////////////////////////////
    cubSt = cublasSetMatrix(Nx, Nx, sizeof(double), Vx, Nx, d_Vx, Nx); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetMatrix(Ny, Ny, sizeof(double), Vy, Ny, d_Vy, Ny); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetMatrix(Nz, Nz, sizeof(double), Vz, Nz, d_Vz, Nz); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetVector(Nx*Ny*Nz, sizeof(double), eig, 1, d_eig, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
////////////////////////////////////////////    
    
    verify_single_col(Nx, Ny, Nz, Vx, Vy, Vz, eig, d_Vx, d_Vy, d_Vz, d_eig);    
    kron_single_col_CPU(Nx, Ny, Nz, Vx, Vy, Vz, eig, reps);    
    kron_single_col_GPU(Nx, Ny, Nz, d_Vx, d_Vy, d_Vz, d_eig, reps);
    
    // multiple column
    verify_multiple_col(Nx, Ny, Nz, ncol, Vx, Vy, Vz, eig, d_Vx, d_Vy, d_Vz, d_eig);    
    kron_multiple_col_CPU(Nx, Ny, Nz, ncol, Vx, Vy, Vz, eig, reps);
    kron_multiple_col_GPU(Nx, Ny, Nz, ncol, d_Vx, d_Vy, d_Vz, d_eig, reps);

    // Periodic complex
    Lap_1D_P_EigenDecomp_complex(Nx, FDn, FDweights_D2_x, Vx_kpt, lambda_x, phase_fac_x);
    Lap_1D_P_EigenDecomp_complex(Ny, FDn, FDweights_D2_y, Vy_kpt, lambda_y, phase_fac_y);
    Lap_1D_P_EigenDecomp_complex(Nz, FDn, FDweights_D2_z, Vz_kpt, lambda_z, phase_fac_z);
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eig);
    
////////////////////////////////////////////
    cubSt = cublasSetMatrix(Nx, Nx, sizeof(cuDoubleComplex), Vx_kpt, Nx, d_Vx_kpt, Nx); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetMatrix(Ny, Ny, sizeof(cuDoubleComplex), Vy_kpt, Ny, d_Vy_kpt, Ny); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetMatrix(Nz, Nz, sizeof(cuDoubleComplex), Vz_kpt, Nz, d_Vz_kpt, Nz); assert(CUBLAS_STATUS_SUCCESS == cubSt);
    cubSt = cublasSetVector(Nx*Ny*Nz, sizeof(double), eig, 1, d_eig, 1); assert(CUBLAS_STATUS_SUCCESS == cubSt);
////////////////////////////////////////////    

    verify_single_col_complex(Nx, Ny, Nz, Vx_kpt, Vy_kpt, Vz_kpt, eig, d_Vx_kpt, d_Vy_kpt, d_Vz_kpt, d_eig);
    kron_single_col_complex_CPU(Nx, Ny, Nz, Vx_kpt, Vy_kpt, Vz_kpt, eig, reps);
    kron_single_col_complex_GPU(Nx, Ny, Nz, d_Vx_kpt, d_Vy_kpt, d_Vz_kpt, d_eig, reps);

    verify_multiple_col_complex(Nx, Ny, Nz, ncol, Vx_kpt, Vy_kpt, Vz_kpt, eig, d_Vx_kpt, d_Vy_kpt, d_Vz_kpt, d_eig);
    kron_multiple_col_complex_CPU(Nx, Ny, Nz, ncol, Vx_kpt, Vy_kpt, Vz_kpt, eig, reps);
    kron_multiple_col_complex_GPU(Nx, Ny, Nz, ncol, d_Vx_kpt, d_Vy_kpt, d_Vz_kpt, d_eig, reps);

    free(FDweights_D2_x);
    free(FDweights_D2_y);
    free(FDweights_D2_z);    
    free(lambda_x);
    free(lambda_y);
    free(lambda_z);
    free(Vx);
    free(Vy);
    free(Vz);
    free(Vx_kpt);
    free(Vy_kpt);
    free(Vz_kpt);
    free(eig);

////////////////////////////////////////
    cuE = cudaFree(d_Vx); assert(cudaSuccess == cuE);
	cuE = cudaFree(d_Vy); assert(cudaSuccess == cuE);
	cuE = cudaFree(d_Vz); assert(cudaSuccess == cuE);
    cuE = cudaFree(d_eig); assert(cudaSuccess == cuE);
////////////////////////////////////////

    return 0;
}




