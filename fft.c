#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>

#include "fft.h"



void fft_solve(int Nx, int Ny, int Nz, double *rhs, double *pois_FFT_const, double *sol)
{
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*(Nx/2+1), Nx/2+1, 1}; 
    int Nd_half = (Nx/2+1)*Ny*Nz;

    double _Complex *rhs_G = (double _Complex *) malloc(sizeof(double _Complex) * Nd_half);    

    MKL_MDFFT_real(rhs, dim_sizes, strides_out, rhs_G);
    
    for (int i = 0; i < Nd_half; i++) {
        rhs_G[i] = creal(rhs_G[i]) * pois_FFT_const[i] 
                    + (cimag(rhs_G[i]) * pois_FFT_const[i]) * I;
    }

    MKL_MDiFFT_real(rhs_G, dim_sizes, strides_out, sol);

    free(rhs_G);
}



void fft_solve_complex(int Nx, int Ny, int Nz, double _Complex *rhs, double *pois_FFT_const, double _Complex *sol, double _Complex *phase_pos, double _Complex *phase_neg)
{
    MKL_LONG dim_sizes[3] = {Nz, Ny, Nx};
    MKL_LONG strides_out[4] = {0, Ny*Nx, Nx, 1}; 
    int Nd = Nx*Ny*Nz;

    double _Complex *rhs_G = (double _Complex *) malloc(sizeof(double _Complex) * Nd);    

    for (int i = 0; i < Nd; i++) {
        rhs[i] *= phase_neg[i];
    }
    MKL_MDFFT(rhs, dim_sizes, strides_out, rhs_G);
    
    for (int i = 0; i < Nd; i++) {
        rhs_G[i] = creal(rhs_G[i]) * pois_FFT_const[i] 
                    + (cimag(rhs_G[i]) * pois_FFT_const[i]) * I;
    }

    MKL_MDiFFT(rhs_G, dim_sizes, strides_out, sol);

    for (int i = 0; i < Nd; i++) {
        rhs[i] *= phase_pos[i];
    }

    free(rhs_G);
}





/**
 * @brief   MKL multi-dimension FFT interface, real to complex, following conjugate even distribution. 
 */
void MKL_MDFFT_real(double *r2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *r2c_3doutput) {
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_REAL, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeForward(my_desc_handle, r2c_3dinput, r2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);
}

/**
 * @brief   MKL multi-dimension FFT interface, complex to complex
 */
void MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput)
{
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_COMPLEX_COMPLEX, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeForward(my_desc_handle, c2c_3dinput, c2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);
}


/**
 * @brief   MKL multi-dimension iFFT interface, complex to real, following conjugate even distribution. 
 */
void MKL_MDiFFT_real(double _Complex *c2r_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_in, double *c2r_3doutput) {
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_REAL, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_INPUT_STRIDES, strides_in);
    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeBackward(my_desc_handle, c2r_3dinput, c2r_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);

    // scale the result to make it the same as definition of IFFT
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    for (int i = 0; i < N; i++) {
        c2r_3doutput[i] /= N;
    }
}

/**
 * @brief   MKL multi-dimension iFFT interface, complex to complex. 
 */
void MKL_MDiFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput)
{
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;
    /********************************************************************/
    
    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_COMPLEX_COMPLEX, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeBackward(my_desc_handle, c2c_3dinput, c2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);

    // scale the result to make it the same as definition of IFFT
    int N = dim_sizes[2]*dim_sizes[1]*dim_sizes[0];
    for (int i = 0; i < N; i++) {
        c2c_3doutput[i] /= N;
    }
}