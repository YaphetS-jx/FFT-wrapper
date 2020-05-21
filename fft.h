#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include "mkl.h"

int bit_reversal(double _Complex *x, int n, double _Complex *y);

int MKL_FFT(double _Complex *c2c_input, int n, double _Complex *c2c_output);

int FFT_radix_2(double _Complex *x, int n, double _Complex *y);

int FFT_Chirp_Z(double _Complex *x, int n, double _Complex *y);

int iFFT_radix_2(double _Complex *x, int n, double _Complex *y);

int iFFT_Chirp_Z(double _Complex *x, int n, double _Complex *y);