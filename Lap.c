#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include <assert.h>
#include <string.h>
#include "mkl.h"

#include "Lap.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))



/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x.
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 * @author  Hua Huang <huangh223@gatech.edu>
 *          Edmond Chow <echow@cc.gatech.edu>
 *
 * @modified by Qimen Xu <qimenxu@gatech.edu>, Mar 2019, Georgia Tech
 *
 * Copyright (c) 2018-2019 Edmond Group at Georgia Tech.
 */
void stencil_3axis_thread_variable_radius(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= radius; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


void stencil_3axis_thread_radius6(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    int i, j, k, jp, kp, r;
    const int shift_ip = x_ex_spos - x_spos;
    for (k = z_spos, kp = z_ex_spos; k < z_epos; k++, kp++)
    {
        for (j = y_spos, jp = y_ex_spos; j < y_epos; j++, jp++)
        {
            int offset = k * stride_z + j * stride_y;
            int offset_ex = kp * stride_z_ex + jp * stride_y_ex;
            #pragma omp simd
            for (i = x_spos; i < x_epos; i++)
            {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_ex = offset_ex + ip;
                double res = coef_0 * x0[idx_ex];
                for (r = 1; r <= 6; r++)
                {
                    int stride_y_r = r * stride_y_ex;
                    int stride_z_r = r * stride_z_ex;
                    double res_x = (x0[idx_ex - r]          + x0[idx_ex + r])          * stencil_coefs[3*r];
                    double res_y = (x0[idx_ex - stride_y_r] + x0[idx_ex + stride_y_r]) * stencil_coefs[3*r+1];
                    double res_z = (x0[idx_ex - stride_z_r] + x0[idx_ex + stride_z_r]) * stencil_coefs[3*r+2];
                    res += res_x + res_y + res_z;
                }
                y[idx] = res + b * (v0[idx] * x0[idx_ex]); 
            }
        }
    }
}


/**
 * @brief   Kernel for calculating y = (a * Lap + b * diag(v0) + c * I) * x.
 *          For the input & output domain, z/x index is the slowest/fastest running index
 *
 * @param x0               : Input domain with extended boundary 
 * @param radius           : Radius of the stencil (radius * 2 = stencil order)
 * @param stride_y         : Distance between y(i, j, k) and y(i, j+1, k)
 * @param stride_y_ex      : Distance between x0(i, j, k) and x0(i, j+1, k)
 * @param stride_z         : Distance between y(i, j, k) and y(i, j, k+1)
 * @param stride_z_ex      : Distance between x0(i, j, k) and x0(i, j, k+1)
 * @param [x_spos, x_epos) : X index range of y that will be computed in this kernel
 * @param [y_spos, y_epos) : Y index range of y that will be computed in this kernel
 * @param [z_spos, z_epos) : Z index range of y that will be computed in this kernel
 * @param x_ex_spos        : X start index in x0 that will be computed in this kernel
 * @param y_ex_spos        : Y start index in x0 that will be computed in this kernel
 * @param z_ex_spos        : Z start index in x0 that will be computed in this kernel
 * @param stencil_coefs    : Stencil coefficients for the stencil points, length radius+1,
 *                           ordered as [x_0 y_0 z_0 x_1 y_1 y_2 ... x_radius y_radius z_radius]
 * @param coef_0           : Stencil coefficient for the center element
 * @param a                : Scaling factor of Lap
 * @param b                : Scaling factor of v0
 * @param c                : Shift constant
 * @param v0               : Values of the diagonal matrix
 * @param beta             : Scaling factor of y
 * @param y (OUT)          : Output domain with original boundary
 *
 */
void stencil_3axis_thread_v2(
    const double *x0,    const int radius, 
    const int stride_y,  const int stride_y_ex, 
    const int stride_z,  const int stride_z_ex,
    const int x_spos,    const int x_epos, 
    const int y_spos,    const int y_epos,
    const int z_spos,    const int z_epos,
    const int x_ex_spos, const int y_ex_spos,  // this allows us to give x as x0 for 
    const int z_ex_spos,                       // calc inner part of Lx
    const double *stencil_coefs, 
    const double coef_0, const double b,   
    const double *v0,    double *y
)
{
    switch (radius)
    {
        case 6:
            stencil_3axis_thread_radius6(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;

        default:
            stencil_3axis_thread_variable_radius(
                x0, radius, stride_y,  stride_y_ex, stride_z, stride_z_ex,
                x_spos, x_epos, y_spos, y_epos, z_spos, z_epos, x_ex_spos, y_ex_spos, z_ex_spos,
                stencil_coefs, coef_0, b, v0, y
            );
            return;
            break;
    }
}



void Lap_SPARC(int FDn, int Nx, int Ny, int Nz, 
    double *FDweights_D2_x, double *FDweights_D2_y, double *FDweights_D2_z,
    const double *X, double *LapX)
{
#define X(i,j,k) X[(i)+(j)*Nx+(k)*NxNy]
#define X_ex(i,j,k) X_ex[(i)+(j)*Nx_ex+(k)*NxNy_ex]

    double *Lap_weights = (double *)malloc(3*(FDn+1)*sizeof(double)); 
    double *Lap_stencil = Lap_weights;
    for (int p = 0; p < FDn + 1; p++)
    {
        (*Lap_stencil++) = FDweights_D2_x[p];
        (*Lap_stencil++) = FDweights_D2_y[p];
        (*Lap_stencil++) = FDweights_D2_z[p];
    }

    double coef_0 = Lap_weights[0] + Lap_weights[1] + Lap_weights[2];
    int NxNy = Nx * Ny;
    int Nx_ex = Nx + 2*FDn;
    int Ny_ex = Ny + 2*FDn;
    int Nz_ex = Nz + 2*FDn;
    int NxNy_ex = Nx_ex * Ny_ex;
    int Nd_ex = NxNy_ex * Nz_ex;

    double *X_ex = (double *) malloc(sizeof(double)* Nd_ex);    

    // copy X into X_ex
    for (int k = 0; k < Nz; k++) {
        int k_ = k + FDn;
        for (int j = 0; j < Ny; j++) {
            int j_ = j + FDn;
            for (int i = 0; i < Nx; i++) {
                int i_ = i + FDn;
                X_ex(i_,j_,k_) = X(i,j,k);
            }
        }
    }

    // 6 edge nodes set 0
    int Nx_out = Nx + FDn;
    int Ny_out = Ny + FDn;
    int Nz_out = Nz + FDn;
    int istart_in[6] = {0,       Nx_out,  FDn,     FDn,     FDn,     FDn   }; 
    int   iend_in[6] = {FDn,     Nx_ex,   Nx_out,  Nx_out,  Nx_out,  Nx_out};
    int jstart_in[6] = {FDn,     FDn,     0,       Ny_out,  FDn,     FDn   };
    int   jend_in[6] = {Ny_out,  Ny_out,  FDn,     Ny_ex,   Ny_out,  Ny_out};
    int kstart_in[6] = {FDn,     FDn,     FDn,     FDn,     0,       Nz_out}; 
    int   kend_in[6] = {Nz_out,  Nz_out,  Nz_out,  Nz_out,  FDn,     Nz_ex };

    for (int nbr_i = 0; nbr_i < 6; nbr_i++) {
        const int kp_s = kstart_in[nbr_i];
        const int kp_e = kend_in  [nbr_i];
        const int jp_s = jstart_in[nbr_i];
        const int jp_e = jend_in  [nbr_i];
        const int ip_s = istart_in[nbr_i];
        const int ip_e = iend_in  [nbr_i];
        for (int kp = kp_s; kp < kp_e; kp++) {
            for (int jp = jp_s; jp < jp_e; jp++) {
                for (int ip = ip_s; ip < ip_e; ip++) {
                    X_ex(ip,jp,kp) = 0.0;
                }
            }
        }
    }    

    stencil_3axis_thread_v2(X_ex, FDn, Nx, Nx_ex, 
                    NxNy, NxNy_ex, 0, Nx,  0, Ny, 0, Nz, 
                    FDn, FDn, FDn, Lap_weights, coef_0, 0, X, LapX);
    
    free(Lap_weights);
    free(X_ex);
}

/**
 * @brief Restrict any function defined on a FD grid to a sub-grid by extracting
 *        the values that fall in the sub-grid.
 *
 *        Note that all the input indices for v_i are relative to the grid owned
 *        by the current process, while the indices for v_o are relative to the
 *        sub-grid in the current process.
 *
 * @param v_i              : Input data on the original grid
 * @param v_o (OUT)        : Output data on the sub-grid
 * @param stride_y_o       : Distance between v_o(i, j, k) and v_o(i, j+1, k)
 * @param stride_y_i       : Distance between v_i(i, j, k) and v_i(i, j+1, k)
 * @param stride_z_o       : Distance between v_o(i, j, k) and v_o(i, j, k+1)
 * @param stride_z_i       : Distance between v_i(i, j, k) and v_i(i, j, k+1)
 * @param [x_spos, x_epos] : X index range of v_o that will be computed
 * @param [y_spos, y_epos] : Y index range of v_o that will be computed
 * @param [z_spos, z_epos] : Z index range of v_o that will be computed
 * @param x_i_spos         : X start index in v_i that will be restricted
 * @param y_i_spos         : Y start index in v_i that will be restricted
 * @param z_i_spos         : Z start index in v_i that will be restricted
 *
 */
void restrict_to_subgrid(
    const double *v_i,    double *v_o,
    const int stride_y_o, const int stride_y_i,
    const int stride_z_o, const int stride_z_i,
    const int x_o_spos,   const int x_o_epos,
    const int y_o_spos,   const int y_o_epos,
    const int z_o_spos,   const int z_o_epos,
    const int x_i_spos,   const int y_i_spos,
    const int z_i_spos
)
{
    const int shift_ip = x_i_spos - x_o_spos;
    const int shift_jp = y_i_spos - y_o_spos;
    const int shift_kp = z_i_spos - z_o_spos;
    for (int k = z_o_spos; k <= z_o_epos; k++) {
        int kp = k + shift_kp;
        for (int j = y_o_spos; j <= y_o_epos; j++) {
            int jp = j + shift_jp;
            int offset = k * stride_z_o + j * stride_y_o;
            int offset_i = kp * stride_z_i + jp * stride_y_i;
            for (int i = x_o_spos; i <= x_o_epos; i++) {
                int ip     = i + shift_ip;
                int idx    = offset + i;
                int idx_i  = offset_i + ip;
                v_o[idx] = v_i[idx_i];
            }
        }
    }
}