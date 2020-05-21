#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "fft.h"
#include <time.h>

int main(int argc, char *argv[]){

    int n = 1<<25-1;
    srand(5);
    clock_t start, finish;

    double _Complex *c2c_input, *mkl_output, *fft_output;
    c2c_input  = (double _Complex *) calloc(n, sizeof(double _Complex));
    mkl_output = (double _Complex *) calloc(n, sizeof(double _Complex));
    fft_output = (double _Complex *) calloc(n, sizeof(double _Complex));
    
    // printf("complex input:\n");
    for (int i = 0; i < n; i++){
        c2c_input[i] = (double)rand()/RAND_MAX + (double)rand()/RAND_MAX * I;
        // printf("%.8f+i*%.8f\n", creal(c2c_input[i]), cimag(c2c_input[i]));
    }
    start = clock();
    MKL_FFT(c2c_input, n, mkl_output);
    finish= clock();

    printf("MKL time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);

    start = clock();
    FFT_Chirp_Z(c2c_input, n, fft_output);
    finish= clock();
    printf("FFT time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);

    // printf("complex output:\n");
    // for (int i = 0; i < n; i++){
        // printf("%.8f+i*%.8f\n", creal(mkl_output[i]), cimag(mkl_output[i]));
    //     printf("%.8f+i*%.8f\n", creal(fft_output[i]), cimag(fft_output[i]));
    // }

    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += (creal(mkl_output[i] - fft_output[i])) * (creal(mkl_output[i] - fft_output[i]));
        sum += (cimag(mkl_output[i] - fft_output[i])) * (cimag(mkl_output[i] - fft_output[i]));
    }

    printf("relative error: %g\n", sqrt(sum)/n);

    free(c2c_input);
    free(mkl_output);
    free(fft_output);

    return 0;
}