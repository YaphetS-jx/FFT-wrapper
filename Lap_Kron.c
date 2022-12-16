#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>
#include <sys/time.h>    // for gettimeofday()

#include "Lap_Kron.h"



void Lap_kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *Lapx)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *rhsTVy = (double *) malloc(sizeof(double) * NxNy);
    double *VxtrhsTVy = Lapx;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = rhsTVy;
    double *VxPTVyt = P;

    double t_s = 0, t_l= 0, t_d = 0;
    struct timeval start, end;

    // P = Lambda .* (Vz' x Vy' x Vx') * rhs
    // gettimeofday( &start, NULL );
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, rhs + k*NxNy, Nx, Vy, Ny, 0.0, rhsTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, rhsTVy, Nx, 0.0, VxtrhsTVy + k*NxNy, Nx);
    }
    // gettimeofday( &end, NULL );
    // t_s += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtrhsTVy, NxNy, Vz, Nz, 0.0, P, NxNy);
    // gettimeofday( &end, NULL );
    // t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // gettimeofday( &start, NULL );
    for (int i = 0; i < Nd; i++) {
        P[i] *= eigenvalue[i];
    }
    // gettimeofday( &end, NULL );
    // t_d += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


    // Lapx = (Vz x Vy x Vx) * P
    // gettimeofday( &start, NULL );
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }
    // gettimeofday( &end, NULL );
    // t_s += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, Lapx, NxNy);
    // gettimeofday( &end, NULL );
    // t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // double t_tot = t_l + t_s + t_d;
    // printf("MM_small takes %.1e ms, MM_large takes %.1e ms, diag takes %.1e ms, percentage %.1f %%, %.1f %%, %.1f %%\n", 
        // t_s, t_l, t_d, t_s/t_tot*100, t_l/t_tot*100, t_d/t_tot*100);

    free(rhsTVy);
    free(P);
}


void Lap_inverse_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *invLapx)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *rhsTVy = (double *) malloc(sizeof(double) * NxNy);
    double *VxtrhsTVy = invLapx;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = rhsTVy;
    double *VxPTVyt = P;

    double t_s = 0, t_l= 0, t_d = 0;
    struct timeval start, end;

    // P = inv(Lambda) .* (Vz' x Vy' x Vx') * rhs
    gettimeofday( &start, NULL );
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, rhs + k*NxNy, Nx, Vy, Ny, 0.0, rhsTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, rhsTVy, Nx, 0.0, VxtrhsTVy + k*NxNy, Nx);
    }
    gettimeofday( &end, NULL );
    t_s += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtrhsTVy, NxNy, Vz, Nz, 0.0, P, NxNy);
    gettimeofday( &end, NULL );
    t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    gettimeofday( &start, NULL );
    for (int i = 0; i < Nd; i++) {
        P[i] /= eigenvalue[i];
    }
    gettimeofday( &end, NULL );
    t_d += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


    // invLapx = inv(Lambda) .* (Vz x Vy x Vx) * P
    gettimeofday( &start, NULL );
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }
    gettimeofday( &end, NULL );
    t_s += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, invLapx, NxNy);
    gettimeofday( &end, NULL );
    t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    double t_tot = t_l + t_s + t_d;
    printf("MM_small takes %.1e ms, MM_large takes %.1e ms, diag takes %.1e ms, percentage %.1f %%, %.1f %%, %.1f %%\n", 
        t_s, t_l, t_d, t_s/t_tot*100, t_l/t_tot*100, t_d/t_tot*100);

    free(rhsTVy);
    free(P);
}


void Lap_inverse_Kron_omp(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *invLapx)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *rhsTVy = (double *) malloc(sizeof(double) * Nd);
    double *VxtrhsTVy = invLapx;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = rhsTVy;
    double *VxPTVyt = P;

    // P = inv(Lambda) .* (Vz' x Vy' x Vx') * rhs
    // #pragma omp parallel for
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, rhs + k*NxNy, Nx, Vy, Ny, 0.0, rhsTVy + k*NxNy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, rhsTVy + k*NxNy, Nx, 0.0, VxtrhsTVy + k*NxNy, Nx);
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtrhsTVy, NxNy, Vz, Nz, 0.0, P, NxNy);

    // #pragma omp parallel for
    for (int i = 0; i < Nd; i++) {
        P[i] /= eigenvalue[i];
    }

    // invLapx = inv(Lambda) .* (Vz x Vy x Vx) * P
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt + k*NxNy, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt + k*NxNy, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, invLapx, NxNy);

    free(rhsTVy);
    free(P);
}