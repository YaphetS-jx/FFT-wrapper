#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"


#ifndef LAP_MATRIX_H
#define LAP_MATRIX_H 

void Lap_3d_Dirichlet(int Nx, int Ny, int Nz, int FDn,
    double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z, double *Lap_3d_D);

void eigval_Lap_3D(int Nx, double *lambda_x, int Ny, double *lambda_y, int Nz, double *lambda_z, double *lambda);

void Lap_1D_D_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *Lap_1D_D, double *V, double *lambda);

void Lap_1D_Dirichlet(int FDn, double *FDweights_D2, int N, double *Lap_1D_D);

void kron(double *A, int mA, int nA, double *B, int mB, int nB, double *C);

void calculate_FDweights_D2(int FDn, double mesh, double *FDweights_D2);

double fract(int n,int k);

void print_matrix(int nrow, int ncol, double *Matrix, char *Name);

#endif 