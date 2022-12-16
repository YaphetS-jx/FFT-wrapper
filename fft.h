#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
// #include "fftw3.h"


#ifndef FFT_H
#define FFT_H 

void MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput);

void MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

void MKL_MDiFFT_real(double _Complex *c2r_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_in, double *c2r_3doutput);

void MKL_MDiFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

// void FFTW_MDFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput);

// void FFTW_MDiFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput);

// void FFTW_MDFFT_real(int *dim_sizes, double *r2c_3dinput, double _Complex *r2c_3doutput);

// void FFTW_MDiFFT_real(int *dim_sizes, double _Complex *c2r_3dinput, double *c2r_3doutput);


#endif 