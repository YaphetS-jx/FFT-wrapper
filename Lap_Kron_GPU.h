#include <stdio.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>


#ifndef LAP_KRON_GPU_H
#define LAP_KRON_GPU_H 

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_Lap_Kron(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
                 double *d_vec, double *d_diag, double *d_out);

void CUDA_Lap_Kron_complex(int Nx, int Ny, int Nz, cuDoubleComplex *d_Vx, cuDoubleComplex *d_Vy, cuDoubleComplex *d_Vz, 
                 cuDoubleComplex *d_VyH, cuDoubleComplex *d_VzH, cuDoubleComplex *d_vec, double *d_diag, cuDoubleComplex *d_out);

#ifdef __cplusplus
}
#endif

#endif 