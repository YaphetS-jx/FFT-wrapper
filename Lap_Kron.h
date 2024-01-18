#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"


#ifndef LAP_KRON_H
#define LAP_KRON_H 

void Lap_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *diag, double *out);

void Lap_Kron_complex(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, double *diag, double _Complex *out);

void Lap_inverse_Kron_omp(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *invLapx);

void Lap_kron_original(int FDn, int Nx, int Ny, int Nz, 
    double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z,
    double *X, double *Lapx);


void Lap_Kron_multicol(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *vec, int ncol, double *diag, double *out);

void f1Tof2(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size);

void f2Tof4(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size);

void f4Tof1(const int Nx, const int Ny, const int Nz, int ncol, 
            void *src, void *dest, const size_t unit_size);

void Lap_Kron_multicol_complex(int Nx, int Ny, int Nz, double _Complex *Vx, double _Complex *Vy, double _Complex *Vz, 
                 double _Complex *VyH, double _Complex *VzH, double _Complex *vec, int ncol, double *diag, double _Complex *out);

#endif