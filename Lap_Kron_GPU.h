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

void CUDA_Lap_Kron_multicol(int Nx, int Ny, int Nz, double *d_Vx, double *d_Vy, double *d_Vz, 
                 double *d_vec, int ncol, double *d_diag, double *d_out);


void f1Tof2_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);

__global__ void f1Tof2_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);

void f2Tof4_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);

__global__ void f2Tof4_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);

void f4Tof1_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);
            
__global__ void f4Tof1_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_src, void *d_dest, const size_t unit_size);

void apply_diagonal_GPU(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_in, double *d_diag, void *d_out, const size_t unit_size);

__global__ void apply_diagonal_kernel(const int Nx, const int Ny, const int Nz, int ncol, 
            void *d_in, double *d_diag, void *d_out, const size_t unit_size);

#ifdef __cplusplus
}
#endif

#endif 