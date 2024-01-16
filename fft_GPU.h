#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef FFT_GPU_H
#define FFT_GPU_H 

#ifdef __cplusplus
extern "C" {
#endif

__global__ void scale_vector(double* a, double scale, int n);

__global__ void GPU_print_kernel(double *a, int n);

__global__ void GPU_print_complex_kernel(cuDoubleComplex *d_vec, int n);

void GPU_print(double *d_vec, int n);

void GPU_print_complex(cuDoubleComplex *d_vec, int n);

void CUDA_MDFFT_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_r2c_3doutput);

void CUDA_MDiFFT_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, cufftDoubleReal *d_c2r_3doutput);

#ifdef __cplusplus
}
#endif

#endif // FFT_GPU_H 