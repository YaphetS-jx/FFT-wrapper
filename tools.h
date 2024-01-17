#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>

#ifndef TOOLS_H
#define TOOLS_H 

#ifdef __cplusplus
extern "C" {
#endif

__global__ void scale_vector(double* a, double scale, int n);

__global__ void scale_vector_complex(cuDoubleComplex* a, double scale, int n);

__global__ void GPU_print_kernel(double *a, int n);

__global__ void GPU_print_complex_kernel(cuDoubleComplex *d_vec, int n);

void GPU_print(double *d_vec, int n);

void GPU_print_complex(cuDoubleComplex *d_vec, int n);

__global__ void Hammond_RR(double* a, double* b, double* c, int n);

__global__ void Hammond_CR(cufftDoubleComplex* a, double* b, cufftDoubleComplex* c, int n);

__global__ void conjugate_vector_kernel(cufftDoubleComplex* a, int n);

void conjugate_vector(cuDoubleComplex* a, int n);

/** @ brief   Copy column-major matrix block
 *
 *  @param unit_size  Size of data element in bytes (double == 8, double _Complex == 16)
 *  @param src_       Pointer to the top-left element of the source matrix 
 *  @param lds        Leading dimension of the source matrix
 *  @param nrow       Number of rows to copy
 *  @param ncol       Number of columns to copy
 *  @param dst_       Pointer to the top-left element of the destination matrix
 *  @param ldd        Leading dimension of the destination matrix
 */
void copy_mat_blk(
    const size_t unit_size, const void *src_, const int lds, 
    const int nrow, const int ncol, void *dst_, const int ldd
);

void rand_vec(double *vec, int len);

double err(double *v1, double *v2, int len);

#ifdef __cplusplus
}
#endif

#endif 