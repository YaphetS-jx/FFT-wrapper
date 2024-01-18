#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"
#include <math.h>
#include <sys/time.h>    // for gettimeofday()
#include <string.h>

#include "Lap_Kron.h"
#include "Lap_Matrix.h"
#include "tools.h"


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

    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Ny, 
                    1.0, vec + k*NxNy, Nx, Vy, Ny, 0.0, vecTVy, Nx);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, vecTVy, Nx, 0.0, VxtvecTVy + k*NxNy, Nx);
    }

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy, Nz, Nz, 
                    1.0, VxtvecTVy, NxNy, Vz, Nz, 0.0, P, NxNy);

    for (int i = 0; i < Nd; i++) {
        P[i] *= diag[i];
    }

    // out = (Vz x Vy x Vx) * P
    for (int k = 0; k < Nz; k++) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx, Ny, Ny, 
                    1.0, P + k*NxNy, Nx, Vy, Ny, 0.0, PTVyt, Nx);

        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny, Nx, 
                    1.0, Vx, Nx, PTVyt, Nx, 0.0, VxPTVyt + k*NxNy, Nx);
    }
    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy, Nz, Nz, 
                    1.0, VxPTVyt, NxNy, Vz, Nz, 0.0, out, NxNy);

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
    double _Complex aplha = 1, beta = 0;

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


void Lap_Kron_multicol(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, int ncol, double *diag, double *out)
{
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    int len = Nd * ncol;
    double *temp = (double *) malloc(sizeof(double) * len);
    
    double *VxTvec, *VxTvecVy, *P, *DP, *VxDP, *VxDpVyT, *res;
    double *FormVxTvec, *FormVxTvecVy, *FormDP, *FormVxDP, *FormVxDpVyT;
    
    VxTvec = VxTvecVy = P = FormDP = FormVxDP = FormVxDpVyT = out;
    FormVxTvec = FormVxTvecVy = DP = VxDP = VxDpVyT = res = temp;

double t_m = 0, t_c = 0;
struct timeval start, end;

    // P = Lambda .* (Vz' x Vy' x Vx') * vec
gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, Nx, Ny*Nz*ncol, Nx, 
                1.0, Vx, Nx, vec, Nx, 0.0, VxTvec, Nx);
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    f1Tof2(Nx, Ny, Nz, ncol, VxTvec, FormVxTvec, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx*Nz*ncol, Ny, Ny, 
                1.0, FormVxTvec, Nx*Nz*ncol, Vy, Ny, 0.0, VxTvecVy, Nx*Nz*ncol);
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    f2Tof4(Nx, Ny, Nz, ncol, VxTvecVy, FormVxTvecVy, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, NxNy*ncol, Nz, Nz, 
                1.0, FormVxTvecVy, NxNy*ncol, Vz, Nz, 0.0, P, NxNy*ncol);
    
    // apply diagonal term
    double *P_ = P, *DP_ = DP;
    for (int nz = 0; nz < Nz; nz++) {
        double *diag_ = diag + nz*Nx*Ny;
        for (int n = 0; n < ncol; n++) {
            for (int i = 0; i < Nx*Ny; i++) {
                DP_[i] = P_[i] * diag_[i];
            }
            P_ += Nx*Ny;
            DP_ += Nx*Ny;
        }
    }
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;

gettimeofday( &start, NULL );
    // out = (Vz x Vy x Vx) * P
    f4Tof1(Nx, Ny, Nz, ncol, DP, FormDP, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, Nx, Ny*Nz*ncol, Nx, 
                1.0, Vx, Nx, FormDP, Nx, 0.0, VxDP, Nx);
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    f1Tof2(Nx, Ny, Nz, ncol, VxDP, FormVxDP, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );    
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, Nx*Nz*ncol, Ny, Ny, 
                1.0, FormVxDP, Nx*Nz*ncol, Vy, Ny, 0.0, VxDpVyT, Nx*Nz*ncol);
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    f2Tof4(Nx, Ny, Nz, ncol, VxDpVyT, FormVxDpVyT, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans, NxNy*ncol, Nz, Nz, 
                1.0, FormVxDpVyT, NxNy*ncol, Vz, Nz, 0.0, res, NxNy*ncol);
gettimeofday( &end, NULL );
t_c += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


gettimeofday( &start, NULL );
    f4Tof1(Nx, Ny, Nz, ncol, res, out, sizeof(double));
gettimeofday( &end, NULL );
t_m += (double)(end.tv_usec - start.tv_usec)/1E3 + (double)(end.tv_sec - start.tv_sec)*1E3;


    free(temp);

// printf("in multicol, data movement takes %.3f ms, blas3 takes %.3f ms, t_m/t_c = %.3f %%\n", t_m, t_c, t_m/t_c*100);
}

void f1Tof2(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size) 
{
    int NxNy = Nx * Ny;
    int lds = Nx;
    int ldd = Nz * ncol * Nx;

    void *src_ = src, *dest_ = dest;
    for (int d1 = 0; d1 < Nz*ncol; d1++) {        
        copy_mat_blk(unit_size, src_, lds, Nx, Ny, dest_, ldd);
        src_ += NxNy * unit_size;
        dest_ += Nx * unit_size;
    }
}


void f2Tof4(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size) 
{
    int NxNy = Nx * Ny;
    int lds = Nz * ncol * Nx;
    int ldd = Nx;

    void *src_ = src;
    for (int d2 = 0; d2 < ncol; d2++) {
        for (int d1 = 0; d1 < Nz; d1++) {
            
            // at (d2,d1) cell
            void *dest_ = dest + unit_size * (d2*NxNy + d1*NxNy*ncol);
            copy_mat_blk(unit_size, src_, lds, Nx, Ny, dest_, ldd);
            src_ += Nx * unit_size;
        }
    }
}

void f4Tof1(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size) 
{
    int NxNy = Nx * Ny;
    int Nd = NxNy * Nz;
    int lds = Nx;
    int ldd = Nx;

    void *src_ = src;
    for (int d2 = 0; d2 < Nz; d2++) {
        for (int d1 = 0; d1 < ncol; d1++) {
            
            void *dest_ = dest + unit_size * (d1*Nd + d2*NxNy);
            copy_mat_blk(unit_size, src_, lds, Nx, Ny, dest_, ldd);
            src_ += NxNy * unit_size;
        }
    }
}

