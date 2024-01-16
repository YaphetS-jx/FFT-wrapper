#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef TEST_H
#define TEST_H 

void Kron_compare_single_col_GPU(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, double *eig,
                    double *d_Vx, double *d_Vy, double *d_Vz, double *d_eig, int reps);


void compare_FFT_CPU_GPU(int Nx, int Ny, int Nz, int reps);

#endif