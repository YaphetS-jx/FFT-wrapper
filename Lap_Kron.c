#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>
#include <sys/time.h>    // for gettimeofday()

#include "Lap_Kron.h"
#include "Lap_Matrix.h"


void Lap_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, double *diag, double *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double *vecTVy = (double *) malloc(sizeof(double) * NxNy);
    double *VxtvecTVy = out;
    double *P = (double *) malloc(sizeof(double) * Nd);
    double *PTVyt = vecTVy;
    double *VxPTVyt = P;

    // double t_s = 0, t_l= 0, t_d = 0;
    // struct timeval start, end;

    // P = Lambda .* (Vz' x Vy' x Vx') * vec
    // gettimeofday( &start, NULL );
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, vec + k*NxNy, Nx, Vy, Ny, 0.0, vecTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, vecTVy, Nx, 0.0, VxtvecTVy + k*NxNy, Nx);
    }
    // gettimeofday( &end, NULL );
    // t_s += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtvecTVy, NxNy, Vz, Nz, 0.0, P, NxNy);
    // gettimeofday( &end, NULL );
    // t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // gettimeofday( &start, NULL );
    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }
    // gettimeofday( &end, NULL );
    // t_d += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


    // out = (Vz x Vy x Vx) * P
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
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, out, NxNy);
    // gettimeofday( &end, NULL );
    // t_l += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

    // double t_tot = t_l + t_s + t_d;
    // printf("MM_small takes %.1e ms, MM_large takes %.1e ms, diag takes %.1e ms, percentage %.1f %%, %.1f %%, %.1f %%\n", 
        // t_s, t_l, t_d, t_s/t_tot*100, t_l/t_tot*100, t_d/t_tot*100);

    free(vecTVy);
    free(P);
}


void Lap_Kron_complex(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    double _Complex *vecTVy = (double _Complex*) malloc(sizeof(double _Complex) * NxNy);
    double _Complex *VxtvecTVy = out;
    double _Complex *P = (double _Complex*) malloc(sizeof(double _Complex) * Nd);
    double _Complex *PTVyt = vecTVy;
    double _Complex *VxPTVyt = P;
    double _Complex aplha = 1, beta = 1;

    // P = Lambda .* (VzH x VyH x VxH) * vec
    for (int k = 0; k < Nz; k++) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    &aplha, vec + k*NxNy, Nx, VyH, Ny, &beta, vecTVy, Nx);

        cblas_zgemm(CblasColMajor, CblasConjTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &aplha, Vx, Nx, vecTVy, Nx, &beta, VxtvecTVy + k*NxNy, Nx);
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    &aplha, VxtvecTVy, NxNy, VzH, Nz, &beta, P, NxNy);

    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }

    // out = (Vz x Vy x Vx) * P    
    for (int k = 0; k < Nz; k++) {
        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    &aplha, P + k*NxNy, Nx, Vy, Ny, &beta, PTVyt, Nx);

        cblas_zgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    &aplha, Vx, Nx, PTVyt, Nx, &beta, VxPTVyt + k*NxNy, Nx);
    }

    cblas_zgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    &aplha, VxPTVyt, NxNy, Vz, Nz, &beta, out, NxNy);

    free(vecTVy);
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


void Lap_kron_original(int FDn, int Nx, int Ny, int Nz, 
    double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z,
    double *X, double *LapX)
{
    double t_reo, t_calc;
    t_reo = t_calc = 0;
    
    int NyNz = Ny * Nz;
    int NxNy = Nx * Ny; 
    double *Xi, *Yi;
    
    // (I3 x I2 x D1) X
    Xi = X; Yi = LapX;
    for (int i = 0; i < NyNz; i++) {
        int ldaXi = 1;
        int ldaYi = 1;
        LapX_1D_Dirichlet(FDn, FDweights_D2_x, Nx, Xi, ldaXi, Yi, ldaYi);
        Xi += Nx;
        Yi += Nx;
    }    
    
    // (I3 x D2 x I1) X
    Xi = X; Yi = LapX;
    for (int j = 0; j < Nz; j++) {
        int shiftj = j * NxNy;
        double *Xi_ = Xi;
        double *Yi_ = Yi;
        for (int i = 0; i < Nx; i++) {
            int ldaXi = Nx;
            int ldaYi = Nx;
            LapX_1D_Dirichlet(FDn, FDweights_D2_y, Ny, Xi_, ldaXi, Yi_, ldaYi);
            Xi_ += 1;
            Yi_ += 1;
        }
        Xi += NxNy;
        Yi += NxNy;
    }    
    
    // (D3 x I2 x I1) X
    Xi = X; Yi = LapX;
    for (int i = 0; i < NxNy; i++) {
        int ldaXi = NxNy;
        int ldaYi = NxNy;
        LapX_1D_Dirichlet(FDn, FDweights_D2_z, Nz, Xi, ldaXi, Yi, ldaYi);
        Xi += 1;
        Yi += 1;
    }
}