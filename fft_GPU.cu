#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>

#include "fft_GPU.h"


#ifdef __cplusplus
extern "C" {
#endif


__global__ void scale_vector(double* a, double scale, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        a[i] *= scale;
    }
}

__global__ void GPU_print_kernel(double *d_vec, int n) 
{
    for (int i = 0; i < n; i++) {
        printf("%20.16f\n", d_vec[i]);
    }
    printf("\n");
}

__global__ void GPU_print_complex_kernel(cuDoubleComplex *d_vec, int n) 
{
    for (int i = 0; i < n; i++) {
        printf("%20.16f + %20.16f\n", cuCreal(d_vec[i]), cuCimag(d_vec[i]));
    }
    printf("\n");
}

void GPU_print(double *d_vec, int n)
{
    GPU_print_kernel<<<1,1>>>(d_vec, n);
}

void GPU_print_complex(cuDoubleComplex *d_vec, int n)
{
    GPU_print_complex_kernel<<<1,1>>>(d_vec, n);
}

void CUDA_MDFFT_real(cufftDoubleReal *d_r2c_3dinput, int *dim_sizes, cufftDoubleComplex *d_r2c_3doutput)
{
    cufftHandle plan_r2c;
    cufftCreate(&plan_r2c);
    cufftPlan3d(&plan_r2c, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_D2Z);
    cufftExecD2Z(plan_r2c, d_r2c_3dinput, d_r2c_3doutput);
    cufftDestroy(plan_r2c);
}

void CUDA_MDiFFT_real(cufftDoubleComplex *d_c2r_3dinput, int *dim_sizes, cufftDoubleReal *d_c2r_3doutput)
{
    // out-place C2R will change input 
    int N_half = (dim_sizes[2]/2+1) * dim_sizes[1] * dim_sizes[0];
    cufftDoubleComplex *d_c2r_3dinput_copy;
    cudaMalloc((void **) &d_c2r_3dinput_copy, sizeof(cufftDoubleComplex) * N_half);
    cudaMemcpy(d_c2r_3dinput_copy, d_c2r_3dinput, sizeof(cufftDoubleComplex)*N_half, cudaMemcpyDeviceToDevice);

    cufftHandle plan_c2r;
    cufftCreate(&plan_c2r);
    cufftPlan3d(&plan_c2r, dim_sizes[0], dim_sizes[1], dim_sizes[2], CUFFT_Z2D);
    cufftExecZ2D(plan_c2r, d_c2r_3dinput_copy, d_c2r_3doutput);
    cufftDestroy(plan_c2r);

    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    int numThreadsPerBlock = 256;
    int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    double scale = 1.0/N;
    scale_vector<<<numBlocks,numThreadsPerBlock>>>(d_c2r_3doutput, scale, N);

    cudaFree(d_c2r_3dinput_copy);
}



#ifdef __cplusplus
}
#endif