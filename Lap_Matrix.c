#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "mkl.h"

#include "Lap_Matrix.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


void Lap_3d_Dirichlet(int Nx, int Ny, int Nz, int FDn,
    double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z, double *Lap_3d_D)
{
#define Lap_3d_D(i,j) Lap_3d_D[(i) + (j)*Nd]
    int NxNy = Nx * Ny;
    int Nd = Nx * Ny * Nz;
    int Nd2 = Nd * Nd;

    int count = 0;
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                // x - dir
                for (int p = max(0, i-FDn); p <= min(Nx-1,i+FDn); p++) {
                    int shift = p - i;
                    Lap_3d_D(count, count + shift) += FDweights_D2_x[abs(shift)];
                }
                // y - dir
                for (int p = max(0, j-FDn); p <= min(Ny-1,j+FDn); p++) {
                    int shift = p - j;
                    Lap_3d_D(count, count + shift * Nx) += FDweights_D2_y[abs(shift)];
                    // printf("count %d, count + shift * Nx %d\n", count, count + shift * Nx);
                }
                // z - dir
                for (int p = max(0, k-FDn); p <= min(Nz-1,k+FDn); p++) {
                    int shift = p - k;
                    Lap_3d_D(count, count + shift * NxNy) += FDweights_D2_z[abs(shift)];
                }
                count ++;
            }
        }
    }
#undef Lap_3d_D
}

void eigval_Lap_3D(int Nx, double *lambda_x, int Ny, double *lambda_y, int Nz, double *lambda_z, double *lambda)
{
    // I3 x I2 x Lx
    int count = 0;
    for (int k = 0; k < Ny*Nz; k++) {
        for (int j = 0; j < Nx; j++) {
            lambda[count ++] = lambda_x[j];
        }
    }

    // I3 x Ly x I1
    count = 0; 
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                lambda[count ++] += lambda_y[j];
            }
        }
    }

    // Lz x I2 x I1
    count = 0; 
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny * Nx; j++) {
            lambda[count ++] += lambda_z[k];
        }
    }
}


void Lap_1D_D_EigenDecomp(int N, int FDn, 
    double mesh, double *FDweights_D2, double *Lap_1D_D, double *V, double *lambda)
{
    calculate_FDweights_D2(FDn, mesh, FDweights_D2);

    if (Lap_1D_D == NULL) Lap_1D_D = V;
    Lap_1D_Dirichlet(FDn, FDweights_D2, N, Lap_1D_D);

    // print_matrix(N, N, Lap_1D_D, "Lap_1D_D");

    if (Lap_1D_D != NULL) 
        memcpy(V, Lap_1D_D, sizeof(double)*N*N);

    int info = LAPACKE_dsyev(LAPACK_COL_MAJOR, 
        'V', 'U', N, V, N, lambda);

    if (info != 0) printf("wrong eigen!");
    
    // print_matrix(N, N, V, "eigenVector");
    // print_matrix(1, N, lambda, "eigenvalue");
}


void Lap_1D_Dirichlet(int FDn, double *FDweights_D2, int N, double *Lap_1D_D)
{
#define  Lap_1D_D(i,j) Lap_1D_D[(j)*N + (i)]

    for (int i = 0; i < N; i++) {
        for (int j = max(0, i-FDn); j <= min(N-1,i+FDn); j++) {
            int shift = abs(j - i);
            Lap_1D_D(i,j) = FDweights_D2[shift];
        }
    }

#undef Lap_1D_D
}



void kron(double *A, int mA, int nA, double *B, int mB, int nB, double *C) 
{
#define  A(i,j) A[(j)*mA + (i)]
#define  B(i,j) B[(j)*mB + (i)]
#define  C(i,j) C[(j)*mC + (i)]

    int mC = mA * mB;
    int nC = nA * nB;
    int count_A = 0;
    
    for (int jA = 0; jA < nA; jA++) {
        for (int iA = 0; iA < mA; iA++) {
            double Aij = A[count_A++];
            int count_B = 0;
            for (int jB = 0; jB < nB; jB++) {
                for (int iB = 0; iB < mB; iB++) {
                    double Bij = B[count_B++];
                    int i_C = iA * mB + iB;
                    int j_C = jA * nB + jB;
                    C(i_C,j_C) = Aij * Bij;
                }
            }
        }
    }

#undef A
#undef B
#undef C
}



void calculate_FDweights_D2(int FDn, double mesh, double *FDweights_D2)
{
    FDweights_D2[0] = 0;
    for (int p = 1; p < FDn + 1; p++) {
        FDweights_D2[0] -= (2.0/(p*p));
        FDweights_D2[p] = (2*(p%2)-1) * 2 * fract(FDn,p) / (p*p);
    }

    for (int p = 0; p < FDn + 1; p++) {
        FDweights_D2[p] /= (mesh * mesh);
    }
}


double fract(int n,int k) {
    int i;
    double Nr=1, Dr=1, val;
    for(i=n-k+1; i<=n; i++)
        Nr*=i;
    for(i=n+1; i<=n+k; i++)
        Dr*=i;
    val = Nr/Dr;
    return (val);
}


void Lap_Dirichlet_1D_CSR(int FDn, double *FDweights_D2, int N, sparse_matrix_t *Lap_D_1d)
{
    assert(N > 2*FDn);

    int NNz = N*(2*FDn+1)-FDn*FDn-FDn;
    double *values = (double *) malloc(sizeof(double) * NNz);
    long long *col_index = (long long *) malloc(sizeof(long long) * NNz);
    long long *row_index = (long long *) malloc(sizeof(long long) * (N+1));

    row_index[0] = 0;
    int count = 0;
    for (int i = 0; i < N; i++) {
        // doing i row
        int nnz_i = 1 + min(FDn,i) + min(N-i-1,FDn);
        for (int j = min(FDn,i); j >= 0; j--) {
            values[count] = FDweights_D2[j];
            col_index[count] = i-j;
            count ++;
        }        

        for (int j = 1; j < min(N-i-1,FDn)+1; j++) {
            values[count] = FDweights_D2[j];
            col_index[count] = i+j;
            count ++;
        }

        row_index[i+1] = count;
    }

    mkl_sparse_d_create_csr( Lap_D_1d, SPARSE_INDEX_BASE_ZERO, N, N, row_index, row_index+1, col_index, values );

    free(values);
    free(col_index);
    free(row_index);
}


void print_matrix(int nrow, int ncol, double *Matrix, char *Name)
{
#define Matrix(i,j) Matrix[(j)*nrow + (i)]

    if (Name != NULL) printf("Matrix %s:\n", Name);

    for (int i = 0; i < nrow; i++) {
        for (int j = 0; j < ncol; j++) {
            printf("%-24.16f  ", Matrix(i,j));
        }
        printf("\n");
    }

#undef Matrix
}