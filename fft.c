#include "fft.h"

int MKL_FFT(double _Complex *c2c_input, int n, double _Complex *c2c_output)
{
    DFTI_DESCRIPTOR_HANDLE my_desc1_handle = NULL;
    MKL_LONG status;

    status = DftiCreateDescriptor(&my_desc1_handle, DFTI_DOUBLE,
                                  DFTI_COMPLEX, 1, n);
    status = DftiSetValue(my_desc1_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiCommitDescriptor(my_desc1_handle);
    status = DftiComputeForward(my_desc1_handle, c2c_input, c2c_output);
    status = DftiFreeDescriptor(&my_desc1_handle);
}

// only working for n is a power of 2
// y=null in-place. otherwise, not-in-place
int bit_reversal(double _Complex *x, int n, double _Complex *y)
{   
    int t, *c, L, i, q, inplace = 0;

    t = round(log(n)/log(2));
    c = (int*) calloc(n, sizeof(int));

    if (y == NULL){
        double _Complex *y = (double _Complex *) calloc(n, sizeof(int));
        inplace = 1;
    }

    c[1] = 1;
    y[0] = x[0];
    y[1] = x[n>>1];

    L = 2;
    q = 2;
    while(L < n) {
        for (i = L; i < (L<<1); i++){
            if(i % 2 == 0){
                c[i] = c[i>>1];
                y[i] = x[c[i]<<(t-q)];
            } else {
                c[i] = c[i>>1] + L;
                y[i] = x[c[i]<<(t-q)];
            }
        }
        L = L<<1;
        q++;
    }

    if (inplace == 1){
        for (i = 0; i < n; i++){
            x[i] = y[i];
        }
        free(y);
    }
    free(c);
    return 0;
}


int FFT_radix_2(double _Complex *x, int n, double _Complex *y)
{
    int t, i, k, N, flag;
    double _Complex *z, *Omega1, *Omega2, top, bot;

    bit_reversal(x, n, y);

    t = round(log(n)/log(2));

    if((1<<t) != n){
        printf("only working when n is a power of 2.\n");
        exit(-1);
    }

    Omega1 = (double _Complex *) calloc(n>>1, sizeof(double _Complex));
    Omega2 = (double _Complex *) calloc(n>>1, sizeof(double _Complex));
    Omega1[0] = 1.0;
    Omega2[0] = 1.0;

    N = 2;
    flag = 0;
    while(N <= n){
        if (flag > 0){
            for (i = 1; i < N/4; i++){
                Omega2[2*i] = Omega1[i];
            }

            for (i = 1; i <= N/4; i+=2){
                Omega2[i] = cos(2 * M_PI / N * i) - sin(2 * M_PI / N * i) * I;
                Omega2[N/2-i] = -1 * conj(Omega2[i]);
            }
        }

        if (flag < 0){
            for (i = 1; i < N/4; i++){
                Omega1[2*i] = Omega2[i];
            }

            for (i = 1; i <= N/4; i+=2){
                Omega1[i] = cos(2 * M_PI / N * i) - sin(2 * M_PI / N * i) * I;
                Omega1[N/2-i] = -1 * conj(Omega1[i]);
            }
        }

        for (k = 0; k < n/N; k++){
            for (i = 0; i < N/2; i++){
                if (flag > 0){
                    top = y[k*N+i] + Omega2[i] * y[k*N+N/2+i];
                    bot = y[k*N+i] - Omega2[i] * y[k*N+N/2+i];    
                } else {
                    top = y[k*N+i] + Omega1[i] * y[k*N+N/2+i];
                    bot = y[k*N+i] - Omega1[i] * y[k*N+N/2+i];    
                }

                y[k*N+i] = top;
                y[k*N+N/2+i] = bot;
            }
        }


        if (flag == 0){
            flag = -1;
        }
        
        flag *= (-1);
        N = N << 1;
    }

    free(Omega1);
    free(Omega2);
}

int iFFT_radix_2(double _Complex *x, int n, double _Complex *y)
{
    int i;
    double _Complex temp;

    for (i = 1; i < (n+1)/2; i++){
        temp = x[i];
        x[i] = x[n-i];
        x[n-i] = temp;
    }

    FFT_radix_2(x, n, y);

    for (i = 0; i < n; i++){
        y[i] /= n;
    }
}


int FFT_Chirp_Z(double _Complex *x, int n, double _Complex *y)
{   
    int N, i;
    double _Complex *omega, *g, *h, *G, *H;

    omega = (double _Complex *) calloc(n, sizeof(double _Complex));

    N = 1<<(int)round(log(n)/log(2));

    if (N == n){
        FFT_radix_2(x, n, y);
        return 0;
    } else {
        while (N < 2*n-1){
            N *= 2;
        }
    }

    g = (double _Complex *) calloc(N, sizeof(double _Complex));
    h = (double _Complex *) calloc(N, sizeof(double _Complex));
    G = (double _Complex *) calloc(N, sizeof(double _Complex));
    H = (double _Complex *) calloc(N, sizeof(double _Complex));

    omega[0] = 1;
    g[0] = x[0];
    for (i = 1; i < n; i++){
        omega[i] = cos(M_PI / n * i * i) - sin(M_PI / n * i * i) * I;
        g[i] = x[i] * omega[i];
    }


    h[0] = omega[0];
    for (i = 1; i < n; i++){
        h[i] = conj(omega[i]);
        h[N-i] = h[i];
    }

    FFT_radix_2(g, N, G);
    FFT_radix_2(h, N, H);

    for (i = 0; i < N; i++){
        G[i] = G[i] * H[i];
    }
    iFFT_radix_2(G, N, H);

    for (i = 0; i < n; i++){
        y[i] = H[i] * omega[i];
    }
    
    free(omega);
    free(g);
    free(h);
    free(G);
    free(H);
}

int iFFT_Chirp_Z(double _Complex *x, int n, double _Complex *y)
{   
    int N, i;
    double _Complex temp;

    N = 1<<(int)round(log(n)/log(2));

    if (N == n){
        iFFT_radix_2(x, n, y);
        return 0;
    }    

    for (i = 1; i < (n+1)/2; i++){
        temp = x[i];
        x[i] = x[n-i];
        x[n-i] = temp;
    }

    FFT_Chirp_Z(x, n, y);

    for (i = 0; i < n; i++){
        y[i] /= n;
    }
}