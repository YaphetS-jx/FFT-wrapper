#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef LAP_KRON_GPU_H
#define LAP_KRON_GPU_H 

#ifdef __cplusplus
extern "C" {
#endif

__global__ void elementWiseMultiply(double* a, double* b, double* c, int n);

void Lap_Kron_CUDA(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
                 double *d_vec, double *d_diag, double *d_out);

#ifdef __cplusplus
}
#endif

#endif 