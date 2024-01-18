#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <cuda_runtime.h>
#include <cuComplex.h>
#include "cublas_v2.h"
#include <cufft.h>

#include "tools.h"


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

__global__ void scale_vector_complex(cuDoubleComplex* a, double scale, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        cuDoubleComplex t;
		t.x = a[i].x * scale;
		t.y = a[i].y * scale;
        a[i] = t;
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

__global__ void Hammond_RR(double* a, double* b, double* c, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] * b[i];
    }
}


__global__ void Hammond_CR(cuDoubleComplex* a, double* b, cuDoubleComplex* c, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
		cuDoubleComplex t;
		t.x = a[i].x * b[i];
		t.y = a[i].y * b[i];
        c[i] = t;
    }
}


__global__ void conjugate_vector_kernel(cuDoubleComplex* a, int n) 
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
		a[i] = cuConj(a[i]);
    }
}

void conjugate_vector(cuDoubleComplex* a, int n) 
{
    int numThreadsPerBlock = 256;
    int numBlocks = (n + numThreadsPerBlock - 1) / numThreadsPerBlock;    

    conjugate_vector_kernel<<<numBlocks,numBlocks>>>(a, n);
}

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
)
{
    if (unit_size == 8)
    {
        size_t col_msize = sizeof(double) * nrow;
        double *src = (double*) src_;
        double *dst = (double*) dst_;
        for (int icol = 0; icol < ncol; icol++)
            memcpy(dst + icol * ldd, src + icol * lds, col_msize);
    }
    if (unit_size == 16)
    {
        size_t col_msize = sizeof(double _Complex) * nrow;
        double _Complex *src = (double _Complex*) src_;
        double _Complex *dst = (double _Complex*) dst_;
        for (int icol = 0; icol < ncol; icol++)
            memcpy(dst + icol * ldd, src + icol * lds, col_msize);
    }
}


__device__ void copy_mat_blk_kernel(
    const size_t unit_size, const void *src_, const int lds, 
    const int nrow, const int ncol, void *dst_, const int ldd
)
{
    // Determine thread x/y indices
    const int ix = blockIdx.x*blockDim.x + threadIdx.x;
    const int iy = blockIdx.y*blockDim.y + threadIdx.y;

    int idx_src = ix + iy * lds;
    int idx_des = ix + iy * ldd;

    if ((ix >= nrow) || (iy >= ncol)) return;

    if (unit_size == sizeof(double)) {
        double *src = (double *) src_;
        double *dst = (double *) dst_;
        dst[idx_des] = src[idx_src];
    }
    if (unit_size == sizeof(cuDoubleComplex)) {
        cuDoubleComplex *src = (cuDoubleComplex *) src_;
        cuDoubleComplex *dst = (cuDoubleComplex *) dst_;
        dst[idx_des] = src[idx_src];
    }


}


void rand_vec(double *vec, int len)
{
    for (int i = 0; i < len; i++) {
        vec[i] = drand48();
    }
}


double err(double *v1, double *v2, int len) 
{
    double err = 0;
    for (int i = 0; i < len; i++) {
        err += fabs(v1[i]-v2[i]);
    }
    return err;
}


#ifdef __cplusplus
}
#endif