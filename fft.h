#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "mkl.h"

int MKL_FFT(double _Complex *c2c_input, int n, double _Complex *c2c_output);

int MKL_FFT_real(double *r2c_input, int n, double *r2c_output);

int MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

int MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput);

//  for complex input case
int bit_reversal(double _Complex *x, int n, double _Complex *y);

int FFT_radix_2(double _Complex *x, int n, double _Complex *y);

int FFT_radix_2_vectors(double _Complex *x, int n, int cols, double _Complex *y);

int FFT_Chirp_Z(double _Complex *x, int n, double _Complex *y);

int FFT_Chirp_Z_vectors(double _Complex *x, int n, int cols, double _Complex *y);

int iFFT_radix_2(double _Complex *x, int n, double _Complex *y);

int iFFT_radix_2_vectors(double _Complex *x, int n, int cols, double _Complex *y);

int iFFT_Chirp_Z(double _Complex *x, int n, double _Complex *y);

int iFFT_Chirp_Z_vectors(double _Complex *x, int n, int cols, double _Complex *y);

int rotate(double _Complex *x, int *size, double _Complex *y);

int FFT_3d(double _Complex *x, int *size, double _Complex *y);

int iFFT_3d(double _Complex *x, int *size, double _Complex *y);

//  for real input case
int FFT_real_2input(double *x1, double *x2, int n, double *y1, double  *y2);

int FFT_real_even(double *x, int n, double *y);

int bit_reversal_real(double *x, int n, double *y);

int FFT_radix_2_real(double *x, int n, double *y);

int FFT_radix_2_vectors_real(double *x, int n, int cols, double *y);

int FFT_Chirp_Z_real(double *x, int n, double *y);

int FFT_Chirp_Z_vectors_real(double *x, int n, int cols, double *y);

int r2c(double *x, int n, double _Complex *y);

int FFT_3d_real(double *x, int *size, double _Complex *y);

int iFFT_radix_2_real(double *x, int n, double *y);

int iFFT_radix_2_vectors_real(double *x, int n, int cols, double *y);

int iFFT_Chirp_Z_real(double *x, int n, double *y);

int iFFT_Chirp_Z_vectors_real(double *x, int n, int cols, double *y);

int iFFT_3d_real(double *x, int *size, double _Complex *y);
