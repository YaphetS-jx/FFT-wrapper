#include <stdio.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#ifndef FFT_GPU_H
#define FFT_GPU_H 

#ifdef __cplusplus
extern "C" {
#endif

void CUDA_fft_solve(int Nx, int Ny, int Nz, double *d_rhs, double *d_pois_FFT_const, double *d_sol);

void CUDA_MDFFT_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_r2c_3doutput);

void CUDA_MDiFFT_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, cufftDoubleReal *d_c2r_3doutput);

void CUDA_MDFFT(cufftDoubleComplex *d_c2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_c2c_3doutput);

void CUDA_MDiFFT(cufftDoubleComplex *d_c2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_c2c_3doutput);

void CUDA_MDFFT_batch_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, int ncol, cufftDoubleComplex *d_r2c_3doutput);

void CUDA_MDiFFT_batch_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, int ncol, cufftDoubleReal *d_c2r_3doutput);

#ifdef __cplusplus
}
#endif

#endif // FFT_GPU_H 