#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef TEST_H
#define TEST_H 

void verify_FFT_CPU_GPU(int Nx, int Ny, int Nz);

void verify_FFT_CPU_GPU_complex(int Nx, int Ny, int Nz);

void FFT_iFFT_CPU(int Nx, int Ny, int Nz, int reps);

void FFT_iFFT_complex_CPU(int Nx, int Ny, int Nz, int reps);

void FFT_iFFT_GPU(int Nx, int Ny, int Nz, int reps);

void FFT_iFFT_complex_GPU(int Nx, int Ny, int Nz, int reps);


void verify_FFT_batch_CPU_GPU(int Nx, int Ny, int Nz, int ncol);




void verify_single_col(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *diag,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag);

void kron_single_col_CPU(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *diag, int reps);

void kron_single_col_GPU(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag, int reps);

void verify_single_col_complex(int Nx, int Ny, int Nz, 
                double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag,
                cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag);

void kron_single_col_complex_CPU(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag, int reps);

void kron_single_col_complex_GPU(int Nx, int Ny, int Nz, cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag, int reps);



void verify_multiple_col(int Nx, int Ny, int Nz, int ncol, double *Vx, double *Vy, double *Vz, double *diag,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag);

void verify_multiple_col_complex(int Nx, int Ny, int Nz, int ncol,
                double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag, 
                cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag);


void kron_multiple_col_CPU(int Nx, int Ny, int Nz, int ncol, double *Vx, double *Vy, double *Vz, double *diag, int reps);

void kron_multiple_col_GPU(int Nx, int Ny, int Nz, int ncol, double *d_Vx, double *d_Vy, double *d_Vz, double *d_diag, int reps);

void kron_multiple_col_complex_CPU(int Nx, int Ny, int Nz, int ncol,
                double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, double *diag, int reps);

void kron_multiple_col_complex_GPU(int Nx, int Ny, int Nz, int ncol, 
        cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, double *d_diag, int reps);
#endif