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
    // status = DftiComputeBackward(my_desc1_handle, c2c_input, c2c_output);
    status = DftiFreeDescriptor(&my_desc1_handle);
}


int MKL_MDFFT(double _Complex *c2c_3dinput, MKL_LONG *dim_sizes, MKL_LONG *strides_out, double _Complex *c2c_3doutput)
{
    DFTI_DESCRIPTOR_HANDLE my_desc_handle = NULL;
    MKL_LONG status;

    status = DftiCreateDescriptor(&my_desc_handle,
                                  DFTI_DOUBLE, DFTI_COMPLEX, 3, dim_sizes);
    status = DftiSetValue(my_desc_handle,
                          DFTI_COMPLEX_COMPLEX, DFTI_COMPLEX_COMPLEX);
    status = DftiSetValue(my_desc_handle, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    status = DftiSetValue(my_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);

    status = DftiCommitDescriptor(my_desc_handle);
    status = DftiComputeForward(my_desc_handle, c2c_3dinput, c2c_3doutput);
    status = DftiFreeDescriptor(&my_desc_handle);
}


// only working for n is a power of 2
// y=null in-place. otherwise, not-in-place
int bit_reversal(double _Complex *x, int n, double _Complex *y)
{   
    int t, *c, L, i, q, inplace = 0;

    t = round(log(n)/log(2));
    c = (int*) calloc(n, sizeof(int));

    if (y == NULL){
        y = (double _Complex *) calloc(n, sizeof(double _Complex));
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

// y=null in-place. otherwise, not-in-place
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
            if (y == NULL){

                for (i = 0; i < N/2; i++){
                    if (flag > 0){
                        top = x[k*N+i] + Omega2[i] * x[k*N+N/2+i];
                        bot = x[k*N+i] - Omega2[i] * x[k*N+N/2+i];    
                    } else {
                        top = x[k*N+i] + Omega1[i] * x[k*N+N/2+i];
                        bot = x[k*N+i] - Omega1[i] * x[k*N+N/2+i];    
                    }

                    x[k*N+i] = top;
                    x[k*N+N/2+i] = bot;
                }

            } else {

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

        }

        if (flag == 0){
            flag = -1;
        }
        
        flag *= (-1);
        N = N << 1;
    }

    free(Omega1);
    free(Omega2);

    return 0;
}

// length(x) = n * cols
int FFT_radix_2_vectors(double _Complex *x, int n, int cols, double _Complex *y)
{
    int t, i, j, k, N, flag;
    double _Complex *z, *Omega1, *Omega2, top, bot;

    if (y == NULL){
        for (i = 0; i < cols; i++){
            bit_reversal(x + i*n, n, NULL);
        }
    } else {
        for (i = 0; i < cols; i++){
            bit_reversal(x + i*n, n, y+i*n);
        }
    }


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
            if (y == NULL){

                for (i = 0; i < N/2; i++){
                    for (j = 0; j < cols; j++){
                        if (flag > 0){
                            top = x[k*N+i + j*n] + Omega2[i] * x[k*N+N/2+i + j*n];
                            bot = x[k*N+i + j*n] - Omega2[i] * x[k*N+N/2+i + j*n];    
                        } else {
                            top = x[k*N+i + j*n] + Omega1[i] * x[k*N+N/2+i + j*n];
                            bot = x[k*N+i + j*n] - Omega1[i] * x[k*N+N/2+i + j*n];    
                        }

                        x[k*N+i + j*n] = top;
                        x[k*N+N/2+i + j*n] = bot;
                    }

                }

            } else {

                for (i = 0; i < N/2; i++){
                    for (j = 0; j < cols; j++){
                        if (flag > 0){
                            top = y[k*N+i + j*n] + Omega2[i] * y[k*N+N/2+i + j*n];
                            bot = y[k*N+i + j*n] - Omega2[i] * y[k*N+N/2+i + j*n];    
                        } else {
                            top = y[k*N+i + j*n] + Omega1[i] * y[k*N+N/2+i + j*n];
                            bot = y[k*N+i + j*n] - Omega1[i] * y[k*N+N/2+i + j*n];    
                        }

                        y[k*N+i + j*n] = top;
                        y[k*N+N/2+i + j*n] = bot;
                    }
                }
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

    return 0;
}



// y=null in-place. otherwise, not-in-place
int iFFT_radix_2(double _Complex *x, int n, double _Complex *y)
{
    int i;
    double _Complex temp;

    if (y == NULL){
        for (i = 1; i < (n+1)/2; i++){
            temp = x[i];
            x[i] = x[n-i];
            x[n-i] = temp;
        }
        FFT_radix_2(x, n, NULL);
        for (i = 0; i < n; i++){
            x[i] /= n;
        }
        return 0;
    }

    y[0] = x[0];
    for (i = 1; i < n; i++){
        y[i] = x[n-i];
    }
    FFT_radix_2(y, n, NULL);
    for (i = 0; i < n; i++){
        y[i] /= n;
    }
    return 0;
}


int iFFT_radix_2_vectors(double _Complex *x, int n, int cols, double _Complex *y)
{
    int i, j;
    double _Complex temp;

    if (y == NULL){
        for (i = 1; i < (n+1)/2; i++){
            for (j = 0; j < cols; j++){
                temp = x[i + j*n];
                x[i + j*n] = x[n-i + j*n];
                x[n-i + j*n] = temp;
            }
        }
        
        FFT_radix_2_vectors(x, n, cols, NULL);

        for (i = 0; i < n * cols; i++){
            x[i] /= n;
        }
        return 0;
    }

    for (i = 0; i < cols; i++){
        y[i * n] = x[i * n];
    }
    
    for (i = 1; i < n; i++){
        for (j = 0; j < cols; j++){
            y[i + j*n] = x[n-i + j*n];
        }
    }
    FFT_radix_2_vectors(y, n, cols, NULL);
    for (i = 0; i < n * cols; i++){
        y[i] /= n;
    }
    return 0;
}

// y=null in-place. otherwise, not-in-place
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

    if (y == NULL){
        for (i = 0; i < n; i++){
            x[i] = H[i] * omega[i];
        }
    } else {
        for (i = 0; i < n; i++){
            y[i] = H[i] * omega[i];
        }
    }
    
    free(omega);
    free(g);
    free(h);
    free(G);
    free(H);

    return 0;
}

int FFT_Chirp_Z_vectors(double _Complex *x, int n, int cols, double _Complex *y)
{   
    int N, i, j;
    double _Complex *omega, *g, *h, *G, *H;

    omega = (double _Complex *) calloc(n, sizeof(double _Complex));

    N = 1<<(int)round(log(n)/log(2));

    if (N == n){
        FFT_radix_2_vectors(x, n, cols, y);
        return 0;
    } else {
        while (N < 2*n-1){
            N *= 2;
        }
    }

    g = (double _Complex *) calloc(N*cols, sizeof(double _Complex));
    h = (double _Complex *) calloc(N, sizeof(double _Complex));
    G = (double _Complex *) calloc(N*cols, sizeof(double _Complex));
    H = (double _Complex *) calloc(N, sizeof(double _Complex));

    omega[0] = 1;
    for (i = 0; i < cols; i++){
        g[i*N] = x[i*n];
    }

    for (i = 1; i < n; i++){
        omega[i] = cos(M_PI / n * i * i) - sin(M_PI / n * i * i) * I;
        for (j = 0; j < cols; j++){
            g[i + j*N] = x[i + j*n] * omega[i];
        }
    }

    h[0] = omega[0];
    for (i = 1; i < n; i++){
        h[i] = conj(omega[i]);
        h[N-i] = h[i];
    }

    FFT_radix_2_vectors(g, N, cols, G);
    FFT_radix_2(h, N, H);

    for (i = 0; i < N; i++){
        for (j = 0; j < cols; j++){
            G[i + j*N] = G[i + j*N] * H[i];
        }
    }

    iFFT_radix_2_vectors(G, N, cols, NULL);

    if (y == NULL){
        for (i = 0; i < n; i++){
            for (j = 0; j < cols; j++){
                x[i + j*n] = G[i + j*N] * omega[i];
            }
        }
    } else {
        for (i = 0; i < n; i++){
            for (j = 0; j < cols; j++){
                y[i + j*n] = G[i + j*N] * omega[i];
            }
        }
    }
    
    free(omega);
    free(g);
    free(h);
    free(G);
    free(H);

    return 0;
}

// y=null in-place. otherwise, not-in-place
int iFFT_Chirp_Z(double _Complex *x, int n, double _Complex *y)
{   
    int N, i;
    double _Complex temp;

    N = 1<<(int)round(log(n)/log(2));

    if (N == n){
        iFFT_radix_2(x, n, y);
        return 0;
    }    

    if (y == NULL){
        for (i = 1; i < (n+1)/2; i++){
            temp = x[i];
            x[i] = x[n-i];
            x[n-i] = temp;
        }
        FFT_Chirp_Z(x, n, NULL);

        for (i = 0; i < n; i++){
            x[i] /= n;
        }
        return 0;
    }

    y[0] = x[0];
    for (i = 1; i < n; i++){
        y[i] = x[n-i];
    }
    FFT_Chirp_Z(y, n, NULL);
    for (i = 0; i < n; i++){
        y[i] /= n;
    }

    return 0;
}

int iFFT_Chirp_Z_vectors(double _Complex *x, int n, int cols, double _Complex *y)
{   
    int N, i, j;
    double _Complex temp;

    N = 1<<(int)round(log(n)/log(2));

    if (N == n){
        iFFT_radix_2_vectors(x, n, cols, y);
        return 0;
    }    

    if (y == NULL){
        for (i = 1; i < (n+1)/2; i++){
            for (j = 0; j < cols; j++){
                temp = x[i + j*n];
                x[i + j*n] = x[n-i + j*n];
                x[n-i + j*n] = temp;
            }
        }
        FFT_Chirp_Z_vectors(x, n, cols, NULL);

        for (i = 0; i < n * cols; i++){
            x[i] /= n;
        }
        return 0;
    }

    for (i = 0; i < cols; i++){
        y[i*n] = x[i*n];    
    }
    
    for (i = 1; i < n; i++){
        for (j = 0; j < cols; j++){
            y[i + j*n] = x[n-i + j*n];
        }
    }
    FFT_Chirp_Z_vectors(y, n, cols, NULL);
    for (i = 0; i < n * cols; i++){
        y[i] /= n;
    }

    return 0;
}

// x[dim[2]][dim[1]][dim[0]] to x[dim[0]][dim[2]][dim[1]]
int rotate(double _Complex *x, int *size, double _Complex *y)
{
    #define x(k,j,i) x[(k)*size[1]*size[0] + (j)*size[0] + i]
    #define y(i,k,j) y[(i)*size[2]*size[1] + (k)*size[1] + j]

    int i, j, k;

    for (k = 0; k < size[2]; k++)
        for (j = 0; j < size[1]; j++)
            for (i = 0; i < size[0]; i++)
                y(i,k,j) = x(k,j,i);

    return 0;    
}

int FFT_3d(double _Complex *x, int *size, double _Complex *y)
{
    int i, j, k, size_temp[3];

    double _Complex *temp;

    temp = (double _Complex *) calloc (size[0]*size[1]*size[2], sizeof(double _Complex));

    FFT_Chirp_Z_vectors(x, size[0], size[1]*size[2], temp);

    rotate(temp, size, y);

    FFT_Chirp_Z_vectors(y, size[1], size[0]*size[2], temp);

    for (i = 0; i < 3; i++){
        size_temp[i] = size[(i+1)%3];
    }
    rotate(temp, size_temp, y);

    FFT_Chirp_Z_vectors(y, size[2], size[0]*size[1], temp);

    for (i = 0; i < 3; i++){
        size_temp[i] = size[(i+2)%3];
    }
    rotate(temp, size_temp, y);

    free(temp);
}



int iFFT_3d(double _Complex *x, int *size, double _Complex *y)
{
    int i, j, k, size_temp[3];

    double _Complex *temp;

    temp = (double _Complex *) calloc (size[0]*size[1]*size[2], sizeof(double _Complex));

    iFFT_Chirp_Z_vectors(x, size[0], size[1]*size[2], temp);

    rotate(temp, size, y);

    iFFT_Chirp_Z_vectors(y, size[1], size[0]*size[2], temp);

    for (i = 0; i < 3; i++){
        size_temp[i] = size[(i+1)%3];
    }
    rotate(temp, size_temp, y);

    iFFT_Chirp_Z_vectors(y, size[2], size[0]*size[1], temp);

    for (i = 0; i < 3; i++){
        size_temp[i] = size[(i+2)%3];
    }
    rotate(temp, size_temp, y);

    free(temp);
}





