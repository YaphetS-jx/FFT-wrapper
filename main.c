#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>
#include <string.h>
#include <omp.h>
#include <sys/time.h>    // for gettimeofday()

#include "fft.h"
#include "Lap.h"
#include "Lap_Matrix.h"
#include "Lap_Kron.h"


void rand_vec(double *vec, int len);

void fft_solve(int Nx, int Ny, int Nz, double *rhs, double *pois_FFT_const, double *sol);

void inv_Lap_test(int Nx, int Ny, int Nz, int reps,
                    double *Vx, double *Vy, double *Vz, double *eigenvalue);

void Lap_test(int Nx, int Ny, int Nz, int reps, int FDn, 
                double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z,
                double *Vx, double *Vy, double *Vz, double *eigenvalue);

int main(int argc, char *argv[]){

    int Nx, Ny, Nz, reps;
    Nx = Ny = Nz = 20;
    reps = 30;
    if( argc == 4 ) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
    } else if (argc == 2) {
        reps = atoi(argv[1]);
    } else if (argc == 5) {
        Nx = atoi(argv[1]);
        Ny = atoi(argv[2]);
        Nz = atoi(argv[3]);
        reps = atoi(argv[4]);
    }
    printf("Nx,Ny,Nz: %d,%d,%d\n", Nx, Ny, Nz);
    printf("reps: %d\n", reps);

    srand(5);    
    /******************************************************/
    /*********          Lap_1D decomposition        *******/
    /******************************************************/

    double dx = 1, dy = 0.2, dz = 0.3;
    double kptTL_x, kptTL_y, kptTL_z;
    kptTL_x = 1; kptTL_y = 2; kptTL_z = 3;
    double _Complex phase_fac_x = cos(kptTL_x) + I * sin(kptTL_x);
    double _Complex phase_fac_y = cos(kptTL_y) + I * sin(kptTL_y);
    double _Complex phase_fac_z = cos(kptTL_z) + I * sin(kptTL_z);
    int FDn = 6, Nd = Nx*Ny*Nz;

    double *FDweights_D2_x = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_y = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_z = (double *) malloc((FDn+1) * sizeof(double));
    double *Vx = (double *) malloc(Nx*Nx * sizeof(double));
    double *Vy = (double *) malloc(Ny*Ny * sizeof(double));
    double *Vz = (double *) malloc(Nz*Nz * sizeof(double));
    double _Complex *Vx_kpt = (double _Complex*) malloc(Nx*Nx * sizeof(double _Complex));
    double _Complex *Vy_kpt = (double _Complex*) malloc(Ny*Ny * sizeof(double _Complex));
    double _Complex *Vz_kpt = (double _Complex*) malloc(Nz*Nz * sizeof(double _Complex));
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    double *eigenvalue = (double *) calloc(sizeof(double),  Nd);

    calculate_FDweights_D2(FDn, dx, FDweights_D2_x);
    calculate_FDweights_D2(FDn, dy, FDweights_D2_y);
    calculate_FDweights_D2(FDn, dz, FDweights_D2_z);


    // Dirichlet
    Lap_1D_P_EigenDecomp(Nx, FDn, FDweights_D2_x, Vx, lambda_x);
    Lap_1D_P_EigenDecomp(Ny, FDn, FDweights_D2_y, Vy, lambda_y);
    Lap_1D_P_EigenDecomp(Nz, FDn, FDweights_D2_z, Vz, lambda_z);
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eigenvalue);


    // Periodic real
    Lap_1D_D_EigenDecomp(Nx, FDn, FDweights_D2_x, Vx, lambda_x);
    Lap_1D_D_EigenDecomp(Ny, FDn, FDweights_D2_y, Vy, lambda_y);
    Lap_1D_D_EigenDecomp(Nz, FDn, FDweights_D2_z, Vz, lambda_z);
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eigenvalue);


    // Periodic complex
    Lap_1D_P_EigenDecomp_complex(Nx, FDn, FDweights_D2_x, Vx_kpt, lambda_x, phase_fac_x);
    Lap_1D_P_EigenDecomp_complex(Ny, FDn, FDweights_D2_y, Vy_kpt, lambda_y, phase_fac_y);
    Lap_1D_P_EigenDecomp_complex(Nz, FDn, FDweights_D2_z, Vz_kpt, lambda_z, phase_fac_z);
    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eigenvalue);
    


    free(FDweights_D2_x);
    free(FDweights_D2_y);
    free(FDweights_D2_z);    
    free(lambda_x);
    free(lambda_y);
    free(lambda_z);
    free(Vx);
    free(Vy);
    free(Vz);
    free(Vx_kpt);
    free(Vy_kpt);
    free(Vz_kpt);
    free(eigenvalue);

    return 0;
}


/******************************************************/
/*********               Lap  test              *******/
/******************************************************/
void Lap_test(int Nx, int Ny, int Nz, int reps, int FDn, 
                double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z,
                double *Vx, double *Vy, double *Vz, double *eigenvalue)
{    
    int Nd = Nx * Ny * Nz;
    double *X = (double *) calloc(sizeof(double), Nd); 
    double *LapX = (double *) calloc(sizeof(double), Nd); 
    double *LapX2 = (double *) calloc(sizeof(double), Nd); 
    double *LapX3 = (double *) calloc(sizeof(double), Nd); 
    
    rand_vec(X, Nd);    

    struct timeval start, end;
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        Lap_SPARC(FDn, Nx, Ny, Nz, FDweights_D2_x, FDweights_D2_y, FDweights_D2_z, X, LapX);
    }
    gettimeofday( &end, NULL );
    double t_SPARC = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // original eigenvalue approach
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {        
        Lap_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, eigenvalue, LapX2);
    }
    gettimeofday( &end, NULL );
    double t_Kron_eig = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;      
    

    // original I3xI2xD1... approach
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {        
        Lap_kron_original(FDn, Nx, Ny, Nz, FDweights_D2_x, FDweights_D2_y, FDweights_D2_z, X, LapX3);        
    }
    gettimeofday( &end, NULL );
    double t_Kron = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


    printf("==============================================\n");
    printf("%d times LapX SPARC way takes   : %.1f ms\n", reps, t_SPARC);
    printf("%d times LapX Kronecker way takes: %.1f ms\n", reps, t_Kron_eig);  
    printf("%d times LapX Kronecker original formulation takes: %.1f ms\n", reps, t_Kron);    
    printf("t_Kron_eig/t_SPARC -1: %.1f %%\tt_Kron/t_SPARC -1: %.1f %%\n", (t_Kron_eig/t_SPARC -1)*100, (t_Kron/t_SPARC -1)*100);

    free(X);    
    free(LapX);
    free(LapX2);
}

/******************************************************/
/*********            inv_Lap  test             *******/
/******************************************************/
void inv_Lap_test(int Nx, int Ny, int Nz, int reps,
                    double *Vx, double *Vy, double *Vz, double *eigenvalue)
{
    int Nd = Nx * Ny * Nz;
    double *pois_FFT_const = (double *) calloc(sizeof(double), (Nx/2+1)*Ny*Nz);
    double *X = (double *) calloc(sizeof(double), Nd); 
    double *invLapX = (double *) calloc(sizeof(double), Nd); 

    rand_vec(pois_FFT_const, (Nx/2+1)*Ny*Nz);

    struct timeval start, end;

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        fft_solve(Nx, Ny, Nz, X, pois_FFT_const, invLapX);
    }
    gettimeofday( &end, NULL );
    double t_FFT = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;
    printf("%d times FFT solver takes   : %.1f ms\n", reps, t_FFT);
    
    double *inv_eig = (double *) malloc(sizeof(double)*Nd);
    for (int i = 0; i < Nd; i++) {
        if (fabs(eigenvalue[i]) < 1e-6) 
            inv_eig[i] = 0;
        else
            inv_eig[i] = 1.0/eigenvalue[i];
    }

    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {        
        Lap_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, inv_eig, invLapX);
    }
    gettimeofday( &end, NULL );
    double t_Kron = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;
    printf("%d times FD_Lap solver takes: %.1f ms\n", reps, t_Kron);    

    printf("t_Kron/t_FFT -1: %.1f %%\n", (t_Kron/t_FFT -1)*100);
    
    free(X);
    free(invLapX);
    free(pois_FFT_const);
    free(inv_eig);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void rand_vec(double *vec, int len)
{
    for (int i = 0; i < len; i++) {
        vec[i] = (double)rand()/RAND_MAX;
    }
}


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


