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

void verify(int FDn, int Nx, int Ny, int Nz, double dx, double dy, double dz);

void rand_vec(double *vec, int len);

void fft_solve(int Nx, int Ny, int Nz, double *rhs, double *pois_FFT_const, double *sol);

void inv_Lap_test(int Nx, int Ny, int Nz, int reps,
                    double *Vx, double *Vy, double *Vz, double *eigenvalue);

void Lap_test(int Nx, int Ny, int Nz, int reps, int FDn,
                    double *Vx, double *Vy, double *Vz, double *eigenvalue, double *Lap_weights);


int main(int argc, char *argv[]){

    int Nx, Ny, Nz, reps;
    Nx = Ny = Nz = 20;
    reps = 100;
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

    // verify(4, 9, 11, 13, 0.1, 0.2, 0.3);

    /******************************************************/
    /*********          Lap_1D decomposition        *******/
    /******************************************************/

    double dx = 1, dy = 0.2, dz = 0.3;
    int FDn = 6, Nd = Nx*Ny*Nz;

    double *FDweights_D2_x = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_y = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_z = (double *) malloc((FDn+1) * sizeof(double));
    double *Vx = (double *) malloc(Nx*Nx * sizeof(double));
    double *Vy = (double *) malloc(Ny*Ny * sizeof(double));
    double *Vz = (double *) malloc(Nz*Nz * sizeof(double));
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    double *eigenvalue = (double *) calloc(sizeof(double),  Nd);

    Lap_1D_D_EigenDecomp(Nx, FDn, dx, FDweights_D2_x, NULL, Vx, lambda_x);
    Lap_1D_D_EigenDecomp(Ny, FDn, dy, FDweights_D2_y, NULL, Vy, lambda_y);
    Lap_1D_D_EigenDecomp(Nz, FDn, dz, FDweights_D2_z, NULL, Vz, lambda_z);
    // print_matrix(Nx, Nx, Dxx, "Dxx");

    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eigenvalue);
    // print_matrix(Nd, 1, eigenvalue, "eigenvalue");

    // inv_Lap_test(Nx, Ny, Nz, reps, Vx, Vy, Vz, eigenvalue);


    double *Lap_weights = (double *)malloc(3*(FDn+1)*sizeof(double)); 
    double *Lap_stencil = Lap_weights;
    for (int p = 0; p < FDn + 1; p++)
    {
        (*Lap_stencil++) = FDweights_D2_x[p];
        (*Lap_stencil++) = FDweights_D2_y[p];
        (*Lap_stencil++) = FDweights_D2_z[p];
    }

    Lap_test(Nx, Ny, Nz, reps, FDn, Vx, Vy, Vz, eigenvalue, Lap_weights);



    free(FDweights_D2_x);
    free(FDweights_D2_y);
    free(FDweights_D2_z);    
    free(lambda_x);
    free(lambda_y);
    free(lambda_z);
    free(Vx);
    free(Vy);
    free(Vz);
    free(eigenvalue);
    free(Lap_weights);

    return 0;
}


/******************************************************/
/*********               Lap  test              *******/
/******************************************************/
void Lap_test(int Nx, int Ny, int Nz, int reps, int FDn,
                    double *Vx, double *Vy, double *Vz, double *eigenvalue, double *Lap_weights)
{
    int Nd = Nx * Ny * Nz;
    int NxNy = Nx*Ny;
    double *pois_FFT_const = (double *) calloc(sizeof(double), (Nx/2+1)*Ny*Nz);
    double *X = (double *) calloc(sizeof(double), Nd); 
    double *LapX = (double *) calloc(sizeof(double), Nd); 
    double *LapX2 = (double *) calloc(sizeof(double), Nd); 
    
    rand_vec(X, Nd);
    rand_vec(pois_FFT_const, (Nx/2+1)*Ny*Nz);
    double coef_0 = Lap_weights[0] + Lap_weights[1] + Lap_weights[2];
    int Nx_ex = Nx + 2*FDn;
    int Ny_ex = Ny + 2*FDn;
    int Nz_ex = Nz + 2*FDn;
    int NxNy_ex = Nx_ex * Ny_ex;
    int Nd_ex = NxNy_ex * Nz_ex;
    double *X_ex = (double *) calloc(sizeof(double), Nd_ex);

    restrict_to_subgrid(X, X_ex, Nx_ex, Nx, NxNy_ex, NxNy, 
        FDn, Nx+FDn-1, FDn, Ny+FDn-1, FDn, Nz+FDn-1, 0, 0, 0);

    struct timeval start, end;
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {
        stencil_3axis_thread_v2(X_ex, FDn, Nx, Nx_ex, 
                    NxNy, NxNy_ex, 0, Nx,  0, Ny, 0, Nz, 
                    FDn, FDn, FDn, Lap_weights, coef_0, 0, X, LapX);
    }
    gettimeofday( &end, NULL );
    double t_SPARC = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;
    printf("%d times LapX SPARC way takes   : %.1f ms\n", reps, t_SPARC);
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {        
        Lap_kron(Nx, Ny, Nz, Vx, Vy, Vz, X, eigenvalue, LapX2);
    }
    gettimeofday( &end, NULL );
    double t_Kron = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;
    printf("%d times LapX Kronecker way takes: %.1f ms\n", reps, t_Kron);    

    printf("t_Kron/t_SPARC -1: %.1f %%\n", (t_Kron/t_SPARC -1)*100);
    
    // check accuracy
    // double err = 0;
    // for (int i = 0; i < Nd; i++) {
    //     err += fabs(LapX2[i] - LapX[i]);
    // }
    // printf("err %e\n", err);


    free(X);
    free(X_ex);
    free(LapX);
    free(LapX2);
    free(pois_FFT_const);
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
    
    gettimeofday( &start, NULL );
    for (int rep = 0; rep < reps; rep++) {        
        Lap_inverse_Kron(Nx, Ny, Nz, Vx, Vy, Vz, X, eigenvalue, invLapX);
    }
    gettimeofday( &end, NULL );
    double t_Kron = (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;
    printf("%d times FD_Lap solver takes: %.1f ms\n", reps, t_Kron);    

    printf("t_Kron/t_FFT -1: %.1f %%\n", (t_Kron/t_FFT -1)*100);
    
    free(X);
    free(invLapX);
    free(pois_FFT_const);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void verify(int FDn, int Nx, int Ny, int Nz, double dx, double dy, double dz)
{
    int Nd = Nx*Ny*Nz;

    double *FDweights_D2_x = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_y = (double *) malloc((FDn+1) * sizeof(double));
    double *FDweights_D2_z = (double *) malloc((FDn+1) * sizeof(double));
    double *Dxx = (double *) malloc(Nx*Nx * sizeof(double));
    double *Dyy = (double *) malloc(Ny*Ny * sizeof(double));
    double *Dzz = (double *) malloc(Nz*Nz * sizeof(double));
    double *Vx = (double *) malloc(Nx*Nx * sizeof(double));
    double *Vy = (double *) malloc(Ny*Ny * sizeof(double));
    double *Vz = (double *) malloc(Nz*Nz * sizeof(double));
    double *lambda_x = (double *) malloc(Nx * sizeof(double));
    double *lambda_y = (double *) malloc(Ny * sizeof(double));
    double *lambda_z = (double *) malloc(Nz * sizeof(double));
    double *eigenvalue = (double *) calloc(sizeof(double),  Nd);
    double *Lap_3d = (double *) calloc(Nd * Nd, sizeof(double));
    double *rhs_r = (double *) calloc(sizeof(double), Nd); 
    double *rhs_r_ex = (double *) calloc(sizeof(double), Nd); 
    double *sol = (double *) calloc(sizeof(double), Nd);     

    Lap_1D_D_EigenDecomp(Nx, FDn, dx, FDweights_D2_x, Dxx, Vx, lambda_x);
    Lap_1D_D_EigenDecomp(Ny, FDn, dy, FDweights_D2_y, Dyy, Vy, lambda_y);
    Lap_1D_D_EigenDecomp(Nz, FDn, dz, FDweights_D2_z, Dzz, Vz, lambda_z);
    // print_matrix(Nx, Nx, Dxx, "Dxx");

    eigval_Lap_3D(Nx, lambda_x, Ny, lambda_y, Nz, lambda_z, eigenvalue);
    // print_matrix(Nd, 1, eigenvalue, "eigenvalue");
    
    Lap_3d_Dirichlet(Nx, Ny, Nz, FDn,
        FDweights_D2_x, FDweights_D2_y, FDweights_D2_z, Lap_3d);
    
    rand_vec(rhs_r, Nd);
    /******************************************************/
    /*********        solve explicitly              *******/
    /******************************************************/

    // make it positive definite
    for (int i = 0; i < Nd * Nd; i++) Lap_3d[i] = -Lap_3d[i];
    for (int i = 0; i < Nd; i++) rhs_r_ex[i] = -rhs_r[i];
    // print_matrix(Nd, 1, rhs_r_ex, "rhs");

    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', Nd, Lap_3d, Nd);
    // print_matrix(Nd, Nd, Lap_3d, "Lap_3d");

    int info  = LAPACKE_dpotrs(LAPACK_COL_MAJOR, 'U', Nd, 1, Lap_3d, Nd, rhs_r_ex, Nd);
    if (info) printf("LAPACKE_dpotrs info %d\n", info);
    // print_matrix(Nd, 1, rhs_r_ex, "sol");

    /******************************************************/
    /*********        solve Kronecker way           *******/
    /******************************************************/
    Lap_inverse_Kron(Nx, Ny, Nz, Vx, Vy, Vz, rhs_r, eigenvalue, sol);

    double err = 0;
    for (int i = 0; i < Nd; i++)
        err += fabs(sol[i] - rhs_r_ex[i]);
    printf("Explicit way V.S. Kronecker way err: %e\n", err);

    Lap_inverse_Kron_omp(Nx, Ny, Nz, Vx, Vy, Vz, rhs_r, eigenvalue, sol);
    err = 0;
    for (int i = 0; i < Nd; i++)
        err += fabs(sol[i] - rhs_r_ex[i]);
    printf("Explicit way V.S. Kronecker_omp way err: %e\n", err);

    free(FDweights_D2_x);
    free(FDweights_D2_y);
    free(FDweights_D2_z);    
    free(Dxx);
    free(Dyy);
    free(Dzz);
    free(lambda_x);
    free(lambda_y);
    free(lambda_z);
    free(Vx);
    free(Vy);
    free(Vz);
    free(sol);
    free(Lap_3d);
    free(rhs_r);    
    free(rhs_r_ex);
    free(eigenvalue);
}


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


