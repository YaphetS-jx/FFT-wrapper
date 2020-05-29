#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include "fft.h"
#include <time.h>

#define d1 3
#define d2 2
#define d3 1

int main(int argc, char *argv[]){

    #define n (1<<20)
    srand(5);
    clock_t start, finish;

// 1d test
    double _Complex c2c_input[n], mkl_output[n], fft_output[n];
    
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

    double sum = 0;
    for (int i = 0; i < n; i++){
        sum += (creal(mkl_output[i] - fft_output[i])) * (creal(mkl_output[i] - fft_output[i]));
        sum += (cimag(mkl_output[i] - fft_output[i])) * (cimag(mkl_output[i] - fft_output[i]));
    }

    printf("relative error: %g\n", sqrt(sum)/n);

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

    printf("relative error: %g\n", sqrt(sum));

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

// // real test1 
    #define rd1 (1<<3)
    double r1[rd1], r2[rd1];
    double rr1[rd1+2], rr2[rd1+2], rr3[rd1+2];

    for(int i = 0; i < rd1; i++){
        r1[i] = (double)rand()/RAND_MAX;
        r2[i] = (double)rand()/RAND_MAX;
        // printf("%.8f\t%.8f\n", r1[i],r2[i]);
    }

    FFT_real_2input(r1, r2, rd1, rr1, rr2);

    // printf("\n\n\n");
    for(int i = 0; i < rd1+2; i++){
        // printf("%.8f\t%.8f\n", rr1[i],rr2[i]);
    }

    FFT_real_even(r1, rd1, rr3);

    sum = 0;
    for (int i = 0; i < rd1 + 2; i++){
        // printf("%.8f\n", rr3[i]);
        sum += (rr1[i] - rr3[i]) * (rr1[i] - rr3[i]);
    }
    printf("relative error: %g\n", sqrt(sum));


// real test2
    double rr4[rd1+2];
    double rr5[rd1+2];
    MKL_FFT_real(r1, rd1, rr4);
    FFT_radix_2_real(r1, rd1, rr5);

    sum = 0;
    for (int i = 0; i < rd1+2; i++){
        sum += (rr4[i] - rr5[i]) * (rr4[i] - rr5[i]);
        // printf("%f, %f\n", rr4[i], rr5[i]);
    }
    printf("relative error: %g\n", sqrt(sum));

    // real test 3
    #define rd2 ((1<<20))

    double z1[rd2], z2[rd2+2], z3[rd2+2];

    for(int i = 0; i < rd2; i++){
        z1[i] = (double)rand()/RAND_MAX;
        // printf("%.8f\n", z1[i]);
    }
    
    start = clock();
    MKL_FFT_real(z1, rd2, z2);
    finish= clock();
    // printf("MKL time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);

    start = clock();
    FFT_Chirp_Z_real(z1, rd2, z3);
    finish= clock();
    // printf("FFT time: %.4f sec\n", (double)(finish- start) / CLOCKS_PER_SEC);
    
    sum = 0;
    for (int i = 0; i < rd2+2; i++){
        sum += (z3[i] - z2[i]) * (z3[i] - z2[i]);
        // printf("%f\n", z2[i]);
    }
    printf("relative error: %g\n", sqrt(sum));

// test real 3d
    // double rinput1[32][100][19];
    // double _Complex routput1[32*100*(19/2+1)];
    // double _Complex routput2[32*100*(19/2+1)];

    // for (int k = 0; k < 32; k++)
    //     for (int j = 0; j < 100; j++)
    //         for (int i = 0; i < 19; i++){

    double rinput1[d3][d2][d1];
    double _Complex routput1[d3*d2*(d1/2+1)];
    double _Complex routput2[d3*d2*(d1/2+1)];
    double _Complex routput3[d3*d2*(d1/2+1)];

    for (int k = 0; k < d3; k++)
        for (int j = 0; j < d2; j++)
            for (int i = 0; i < d1; i++){
                rinput1[k][j][i] = (double)rand()/RAND_MAX;
                // printf("%.8f\n", rinput1[k][j][i]);
            }

            // printf("\n\n\n");
    FFT_3d_real(rinput1[0][0], size, routput1);
    iFFT_3d_real(rinput1[0][0], size, routput3);

    strides_out[1] = d2 * (d1/2+1);
    strides_out[2] = d1/2+1;

    MKL_MDFFT_real(rinput1[0][0], dim_sizes, strides_out, routput2);
    
    sum = 0;
    for (int i = 0; i < (d1/2+1)*d2*d3; i++){
        sum += creal(routput1[i] - routput2[i]) * creal(routput1[i] - routput2[i]);
        sum += cimag(routput1[i] - routput2[i]) * cimag(routput1[i] - routput2[i]);
        // printf("%.8f+i*%.8f\n", creal(routput3[i]), cimag(routput3[i]));
    }
    printf("relative error: %g\n", sqrt(sum));

// test real vectors
    double rv1[d3*d2*(d1/2*2+2)];
    double rv2[d3*d2*(d1/2*2+2)];
    double *in = rinput1[0][0];
    for (int k = 0; k < d3; k++)
        for (int j = 0; j < d2; j++){
            MKL_FFT_real(in + k*d2*d1 + j*d1, d1, rv1 + k*d2*(d1/2*2+2) + j*(d1/2*2+2));
        }

    FFT_Chirp_Z_vectors_real(in, d1, d2*d3, rv2);

    sum = 0;
    for (int i = 0; i < (d1/2*2+2)*d2*d3; i++){
        sum += (rv2[i]-rv1[i])*(rv2[i]-rv1[i]);
        printf("%.8f\t %.8f\n", rv1[i], rv2[i]);
    }
    printf("relative error: %g\n", sqrt(sum));




    return 0;
}

