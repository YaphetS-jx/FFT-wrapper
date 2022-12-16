#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>

#include "fft.h"

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


// void FFTW_MDFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput) {
//     fftw_complex *in, *out;
//     fftw_plan p;
//     int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2];
//     p = fftw_plan_dft(3, dim_sizes, c2c_3dinput, c2c_3doutput, FFTW_FORWARD, FFTW_ESTIMATE);
//     fftw_execute(p);
//     fftw_destroy_plan(p);
// }

// void FFTW_MDiFFT(int *dim_sizes, double _Complex *c2c_3dinput, double _Complex *c2c_3doutput) {
//     fftw_complex *in, *out;
//     fftw_plan p;
//     int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2], i;
//     p = fftw_plan_dft(3, dim_sizes, c2c_3dinput, c2c_3doutput, FFTW_BACKWARD, FFTW_ESTIMATE);
//     fftw_execute(p);
//     fftw_destroy_plan(p);
//     for (i = 0; i < N; i++)
//         c2c_3doutput[i] /= N;
// }


// void FFTW_MDFFT_real(int *dim_sizes, double *r2c_3dinput, double _Complex *r2c_3doutput) {
//     fftw_complex *in, *out;
//     fftw_plan p;
//     int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2];
//     p = fftw_plan_dft_r2c(3, dim_sizes, r2c_3dinput, r2c_3doutput, FFTW_ESTIMATE);
//     fftw_execute(p);
//     fftw_destroy_plan(p);
// }

// void FFTW_MDiFFT_real(int *dim_sizes, double _Complex *c2r_3dinput, double *c2r_3doutput) {
//     fftw_complex *in, *out;
//     fftw_plan p;
//     int N = dim_sizes[0] * dim_sizes[1] * dim_sizes[2], i;
//     p = fftw_plan_dft_c2r(3, dim_sizes, c2r_3dinput, c2r_3doutput, FFTW_ESTIMATE);
//     fftw_execute(p);
//     fftw_destroy_plan(p);
//     for (i = 0; i < N; i++)
//         c2r_3doutput[i] /= N;
// }


