#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include "mkl.h"


#ifndef LAP_KRON_H
#define LAP_KRON_H 

void Lap_kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *Lapx);

void Lap_inverse_Kron(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *invLapx);

void Lap_inverse_Kron_omp(int Nx, int Ny, int Nz, double *Vx, double *Vy, double *Vz, 
                 double *rhs, double *eigenvalue, double *invLapx);
                 
#endif