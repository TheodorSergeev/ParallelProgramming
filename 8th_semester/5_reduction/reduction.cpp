/*
    Fedor Sergeev (2020) copyright

    Implementation of the tridiagonal matrix algorithm and cyclic reduction algorithm.
    Based on code by Nikolay Khokhlov.

    Build: make
    Run: mpiexec -np 3 ./main 4
    (here 3 is the number of processes, 4 is the size of a matrix that will be used)
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <mpi.h>


const int MSG_TAG = 123;
#define SEND_PAR MSG_TAG, MPI_COMM_WORLD
#define RECV_PAR MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE

// Technical functions
int int_pow(int base, int exp); // Compute base in power exp for integers


// dot product (a,b)
double dot(const double* a, const double* b, const int n);

// c = alpha * a + beta * b
// (here alpha, beta are scalars; a, b are vectors of size n)
void addv(const double alpha, const double* a, const double beta, const double *b, const int n, double* c);

// d = A * x
// (here d0, d1, d_1 are diagonals of the tridiagonal matrix; b is the right hand side vector)
void matvec(const double* d0, const double* d1, const double* d_1, const double* x, const int n, double* b);

// b = a
// (here aa, b are vectors of size n)
void copyv(const double* a, const int n, double* b);

// x = A^-1 * b
// (here d0, d1, d_1 are diagonals of the tridiagonal matrix, b - right hand side vector)
void tridiag_alg(const double* d0, const double* d1, const double* d_1, const double* b, const int n, double* x);

// Generate tridiagonal matrix
// (here d0, d1, d_1 are diagonals of the tridiagonal matrix, b - right hand side vector)
// d0 is the central diagonal, d1 is the right diagonal, d_1 is the left diagonal
void gen_matrix(const int n, double** d0, double** d1, double** d_1, double** b);

enum ALG_ID {TRIDIAGONAL, CYCLE_REDUCTION};
double lin_sys_solver(int matr_size, ALG_ID algorithm);


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    if (argc < 2)
    {
        printf("Missing matrix size command line parameter. Usage: %s N\n", argv[0]);
        exit(1);
    }

    int matr_size = atoi(argv[1]);
    double tolerance = 0.0;

    //tolerance = lin_sys_solver(matr_size, TRIDIAGONAL);
    //printf("Tridiagonal algorithm: ||A*x-b|| %e\n", tolerance);

    tolerance = lin_sys_solver(matr_size, CYCLE_REDUCTION);
    printf("Cycle reduction algorithm: ||A*x-b|| %e\n", tolerance);

    MPI_Finalize();
    return 0;
}


int int_pow(int base, int exp)
{
    int result = 1;

    while (exp)
    {
        if (exp % 2)
        {
           result *= base;
        }
        exp /= 2;
        base *= base;
    }
    return result;
}

double dot(const double* a, const double* b, const int n)
{
    double res = 0.0;

    for (int i = 0; i < n; i++)
    {
        res += a[i] * b[i];
    }

    return res;
}

void addv(const double alpha, const double* a, const double beta, const double* b, const int n, double* c)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = alpha * a[i] + beta * b[i];
    }
}

void matvec(const double* d0, const double* d1, const double* d_1, const double* x, const int n, double* b)
{
    for (int i = 1; i < n - 1; i++)
    {
        b[i] = d0[i] * x[i] + d1[i] * x[i + 1] + d_1[i - 1] * x[i - 1];
    }

    b[0]   = d0[0] * x[0] + d1[0] * x[1];
    b[n - 1] = d0[n - 1] * x[n - 1] + d_1[n - 2] * x[n - 2];
}

void copyv(const double *a, const int n, double *b)
{
    memcpy(b, a, sizeof(double) * n);
}

void gen_matrix(const int n, double** d0, double** d1, double** d_1, double** b, int pad)
{
    *d0  = (double*) calloc(pad + n,     sizeof(double));
    *d1  = (double*) calloc(pad + n - 1, sizeof(double));
    *d_1 = (double*) calloc(pad + n - 1, sizeof(double));
    *b   = (double*) calloc(pad + n,     sizeof(double));

    // fill diogonal vectors
    for(int i = 0; i < n - 1; i++)
    {
        (*d0) [i] = 2.0;
        (*d_1)[i] = 1.0;
        (*d1) [i] = 1.0;
        (*b)  [i] = 2.0;
    }

    (*d0)[n - 1] = 2.0;
    (*b) [n - 1] = 2.0;

    for (int i = n; i < pad + n; i++)
    {
        (*d0) [i] = 1.0;
    }
}

void tridiag_alg(const double *d0, const double *d1, const double *d_1, const double *b, const int n, double *x)
{
    copyv(b, n, x);
    double* c = (double*) calloc(n - 1, sizeof(double));
    copyv(d1, n - 1, c);
    c[0] = c[0] / d0[0];
    x[0] = x[0] / d0[0];

    for (int i = 1; i < n; i++)
    {
        double m = 1.0 / (d0[i] - d_1[i - 1] * c[i - 1]);
        x[i] = (x[i] - d_1[i - 1] * x[i - 1]) * m;

        if (i < n - 1)
        {
            c[i] *= m;
        }
    }

    printf("%d %lf\n", n - 1, x[n - 1]);
    for (int i = n - 2; i >= 0; i--)
    {
        x[i] -= c[i] * x[i + 1];
        printf("%d %lf\n", i, x[i]);

    }
}

void cyc_red_alg(double *d0, double *d1, double *d_1, double *rhs_vec, const int matr_size, double *x)
{
    assert(matr_size % 2 == 0); // sanity check
    // (matr_size actually must be badded to a power of 2)

    double* f_arr = (double*) calloc(matr_size, sizeof(double));

    double* a = d_1;
    double* b = d0;
    double* c = d1;

    double alpha = 0.0, beta = 0.0;

    for (int i = 0; i <= matr_size / 2; i *= 2)
    {
        beta = - c[0] / b[i];
        b[0] += beta * a[i];

        f_arr[0] += beta * f_arr[i];
        c[0] = beta * c[i];

        alpha = - a[matr_size - 1] / b[matr_size - 1 - i];
        b[matr_size - 1] += alpha * c[matr_size - 1 - i];

        f_arr[matr_size - 1] += alpha * f_arr[matr_size - 1 - i];
        a[matr_size - 1] = alpha * a[matr_size - 1 - i];

        for (int j = 2 * i; j <= matr_size - 2 * i; j += 2 * i)
        {
            alpha = - a[j] / b[j - i];
            beta  = - c[j] / b[j + i];

            b[j]  = b[j] + alpha * c[j - i] + beta * a[j + i];
            f_arr[j] = f_arr[j]  + alpha * f_arr[j - i] + beta * f_arr[j + i];

            a[j] = alpha * a[j - i];
            c[j] = beta  * c[j + i];
        }
    }

    double denom = b[0] * b[matr_size - 1] - a[0] * c[matr_size - 1];
    assert(denom != 0.0);

    x[0] = (f_arr[0] * b[matr_size - 1] - c[matr_size - 1] * f_arr[matr_size - 1]) / denom;
    x[matr_size - 1] = (b[0] * f_arr[matr_size - 1] - a[0] * f_arr[0]) / denom;

    for (int s = matr_size / 2; s >= 1; s /= 2)
    {
        for (int j = s; j <= matr_size - s; j += 2 * s)
        {
            x[j] = (f_arr[j] - a[j] * x[j - s] - c[j] * x[j + s]) / b[j];
            printf("%d %lf\n", j, x[j]);
        }
    }

    printf("%d %lf\n", 0, x[0]);
    printf("%d %lf\n", matr_size - 1, x[matr_size - 1]);

    free(f_arr);
    a = NULL;
    b = NULL;
    c = NULL;
}


double lin_sys_solver(int matr_size, ALG_ID algorithm)
{
    double* x = NULL;
    double *d0, *d1, *d_1, *rhs_vec;

    const int NO_PAD = 0;

    int padded_matr_size = matr_size;

    switch (algorithm)
    {
        case TRIDIAGONAL:
            gen_matrix(matr_size, &d0, &d1, &d_1, &rhs_vec, NO_PAD);
            x = (double*) calloc(matr_size, sizeof(double));
            tridiag_alg(d0, d1, d_1, rhs_vec, matr_size, x);
            break;

        case CYCLE_REDUCTION:
            // closest to matr_size power of 2 (that is greater than matr_size)
            const int exp = (int) ceil(log2((double) matr_size));
            padded_matr_size = int_pow(2, exp);
            printf("padded size %d\n", padded_matr_size);
            gen_matrix(matr_size, &d0, &d1, &d_1, &rhs_vec, padded_matr_size);
            x = (double*) calloc(padded_matr_size, sizeof(double));
            cyc_red_alg(d0, d1, d_1, rhs_vec, padded_matr_size, x);
            break;
    }

    double *tol_vec = (double*) calloc(padded_matr_size, sizeof(double));
    matvec(d0, d1, d_1, x, padded_matr_size, tol_vec);
    addv(1.0, tol_vec, -1.0, rhs_vec, padded_matr_size, tol_vec);

    double tol_norm = sqrt(dot(tol_vec, tol_vec, padded_matr_size)); // ||Ax-b||

    assert(x != NULL);
    free(x);
    free(rhs_vec);
    free(d0);
    free(d1);
    free(d_1);

    return tol_norm;
}