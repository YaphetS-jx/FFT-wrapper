#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "mkl.h"

int bit_reversal(double _Complex *x, int n, double _Complex *y);

int MKL_FFT(double _Complex *c2c_input, int n, double _Complex *c2c_output);

int MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

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