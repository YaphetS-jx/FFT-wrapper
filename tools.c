#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "tools.h"


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