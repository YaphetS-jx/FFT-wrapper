#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "fft.h"
#include <time.h>

#define d1 4
#define d2 3
#define d3 2

int main(int argc, char *argv[]){

    int n = 9973;
    srand(5);
    clock_t start, finish;

// 1d test
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

    // printf("MKL time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);

    start = clock();
    FFT_Chirp_Z(c2c_input, n, fft_output);
    finish= clock();
    // printf("FFT time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);

    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += (creal(mkl_output[i] - fft_output[i])) * (creal(mkl_output[i] - fft_output[i]));
        sum += (cimag(mkl_output[i] - fft_output[i])) * (cimag(mkl_output[i] - fft_output[i]));
    }

    // printf("relative error: %g\n", sqrt(sum)/n);

    free(c2c_input);
    free(mkl_output);
    free(fft_output);

// n vectors test
    double _Complex input1[d3][d2][d1];
    double _Complex output1[d3*d2*d1];
    double _Complex output2[d3*d2*d1];

    for (int k = 0; k < d3; k++)
        for (int j = 0; j < d2; j++)
            for (int i = 0; i < d1; i++){
                input1[k][j][i] = (double)rand()/RAND_MAX + (double)rand()/RAND_MAX * I;
                // printf("%.8f+i*%.8f\n", creal(input1[k][j][i]), cimag(input1[k][j][i]));
            }

    double _Complex *input = input1[0][0];
    for (int k = 0; k < d3; k++)
        for (int j = 0; j < d2; j++){
            MKL_FFT(input + k*d2*d1 + j*d1, d1, output1 + k*d2*d1 + j*d1);
        }
    // printf("\n\n\n\n");
    
    FFT_Chirp_Z_vectors(input, d1, d2*d3, output2);
    // iFFT_radix_2_vectors(input, d1, d2*d3, output2);

    for (int i = 0; i < d1*d2*d3; i++){
        // printf("%.8f+i*%.8f\n", creal(output2[i]), cimag(output2[i]));
    }
    sum = 0;
    for (int i = 0; i < d1*d2*d3; i++){
        sum += (creal(output1[i] - output2[i])) * (creal(output1[i] - output2[i]));
        sum += (cimag(output1[i] - output2[i])) * (cimag(output1[i] - output2[i]));
    }

    // printf("relative error: %g\n", sqrt(sum));

// test rotate
    // for (int k = 0; k < d3; k++)
    //     for (int j = 0; j < d2; j++)
    //         for (int i = 0; i < d1; i++){
    //             input1[k][j][i] = i+j*d1+k*d1*d2;
    //             printf("%.0f\n", input1[k][j][i]);
    //         }

    // int size[3] = {d1,d2,d3};
    // rotate(input1[0][0], size, output1);
    // size[0] = d2; size[1] = d3; size[2] = d1;
    // rotate(output1, size, input1[0][0]);
    // size[0] = d3; size[1] = d1; size[2] = d2;
    // rotate(input1[0][0], size, output1);

    // for (int i = 0; i < d1*d2*d3; i++){
    //     printf("%.0f\n", output1[i]);
    // }

// test 3d fft
    int size[3] = {d1,d2,d3};
    FFT_3d(input, size, output2);
    // for (int i = 0; i < d1*d2*d3; i++){
    //     printf("%.8f+i*%.8f\n", creal(output2[i]), cimag(output2[i]));
    // }

    MKL_LONG dim_sizes[3] = {d3, d2, d1};
    MKL_LONG strides_out[4] = {0, d2*d1*1, d1*1, 1}; 

    MKL_MDFFT(input, dim_sizes, strides_out, output1);

    sum = 0;
    for (int i = 0; i < d1*d2*d3; i++){
        sum += (creal(output1[i] - output2[i])) * (creal(output1[i] - output2[i]));
        sum += (cimag(output1[i] - output2[i])) * (cimag(output1[i] - output2[i]));
    }
    printf("relative error: %g\n", sqrt(sum));

    return 0;
}





