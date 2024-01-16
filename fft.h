#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
// #include "fftw3.h"


#ifndef FFT_H
#define FFT_H 

void fft_solve(int Nx, int Ny, int Nz, double *rhs, double *pois_FFT_const, double *sol);

void fft_solve_complex(int Nx, int Ny, int Nz, double _Complex *rhs, double *pois_FFT_const, double _Complex *sol, double _Complex *phase_pos, double _Complex *phase_neg);

void MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput);

void MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

void MKL_MDiFFT_real(double _Complex *c2r_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_in, double *c2r_3doutput);

void MKL_MDiFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput);

#endif 