/*
    Fedor Sergeev (2020) copyright

    Build and run: make
    Alternative build: mpiCC broadcast_reduce_timing.cc -o main
    Alternative run: mpiexec -np 3 ./main
    (here 3 is the number of processes)

    syntax checked with vera++
*/

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>


#define SEND_PAR MSG_TAG, MPI_COMM_WORLD
#define RECV_PAR MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
const int MSG_TAG = 123;


// Broadcast and reduce roots
const int BCAST_ROOT = 0;
const int REDUCE_ROOT = 0;

// Perform broadcast using linear scheme (recv_arr to all processes from BCAST_ROOT)
void bcast_linear    (double* recv_arr, const int arr_size);

// Perform reduce using linear scheme (recv_arr to all processes from BCAST_ROOT, result stores in RECV_ARR)
void reduce_linear   (double* send_arr, double* recv_arr, const int arr_size);

// Integer power function ($base^exp$)
int int_pow(int base, int exp);

// Perform broadcast using hypercube scheme (recv_arr to all processes from BCAST_ROOT)
void bcast_hypercube (int rank, int world_size, double* send_arr, double* recv_arr, const int arr_size);

// Perform reduce using hypercube scheme (reduces send_arr to the REDUCE_ROOT process, result stores in RECV_ARR)
void reduce_hypercube(int rank, int world_size, double* send_arr, double* recv_arr, const int arr_size);

// Calculate linear/hypercube bcast/reduce execution time on current world_size and write it to file fname
int timer_run(int rank, int world_size, const char* fname, double* send_arr, double* recv_arr, int arr_size);


int main(int argc, char** argv)
{
    if (argc != 1)
    {
        printf("Expected the number of command line arguments to be "
               "argc=1, got %d arguments instead. Exit\n", argc);
        return 1;
    }


    int rank = 0, world_size = 0;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // double == 4 bytes, we would like to pass ~10-100 Mb
    // => we choose 5e6 doubles == 40 Mb

    const int SEND_RECV_ARR_SIZE = 5e6;
    double* SEND_ARR = (double*) calloc(SEND_RECV_ARR_SIZE, sizeof(double));
    double* RECV_ARR = (double*) calloc(SEND_RECV_ARR_SIZE, sizeof(double));

    if (SEND_ARR == nullptr || RECV_ARR == nullptr)
    {
        printf("Couldn't allocate arrays\n");
        return 1;
    }

    // fill arrays with some values so we could test the functions
    for (int i = 0; i < SEND_RECV_ARR_SIZE; ++i)
    {
        SEND_ARR[i] = (double) i;

        if (rank == BCAST_ROOT)
            RECV_ARR[i] = (double) i;
        else
            RECV_ARR[i] = -123.0;
    }

    enum MODES
    {
        LIN,   // test linear bcast/reduce
        CUB,   // test hypercube bcast/reduce
        TIMER  // calculate linear/hypercube bcast/reduce execution time
    };

    const int mode = TIMER;

    switch (mode)
    {
        case LIN:
            bcast_linear (RECV_ARR, SEND_RECV_ARR_SIZE);
            printf("BCAST 4th el (should be equal): %lf\n", RECV_ARR[4]);
            reduce_linear(SEND_ARR, RECV_ARR, SEND_RECV_ARR_SIZE);
            printf("REDUCE 4th el (should be *world_size): %lf\n", RECV_ARR[4]);
            break;

        case CUB:
            bcast_hypercube (rank, world_size, SEND_ARR, RECV_ARR, SEND_RECV_ARR_SIZE);
            printf("CUBE BCAST 4th el (should be equal): %lf\n", RECV_ARR[4]);
            reduce_hypercube(rank, world_size, SEND_ARR, RECV_ARR, SEND_RECV_ARR_SIZE);
            printf("CUBE REDUCE 4th el (should be *world_size): %lf\n", SEND_ARR[4]);

        case TIMER:
            const char* fname = "reduce_bcast_speed.txt";
            timer_run(rank, world_size, fname, SEND_ARR, RECV_ARR, SEND_RECV_ARR_SIZE);
    }

    free(SEND_ARR);
    free(RECV_ARR);
    SEND_ARR = nullptr;
    RECV_ARR = nullptr;

    MPI_Finalize();
    return 0;
}


void reduce_linear(double* send_arr, double* recv_arr, const int arr_size)
{
    assert (send_arr != nullptr && recv_arr != nullptr && arr_size > 0);
    MPI_Reduce(send_arr, recv_arr, arr_size, MPI_DOUBLE, MPI_SUM, REDUCE_ROOT, MPI_COMM_WORLD);
}

void bcast_linear(double* recv_arr, const int arr_size)
{
    assert (recv_arr != nullptr && arr_size > 0);
    MPI_Bcast(recv_arr, arr_size, MPI_DOUBLE, BCAST_ROOT, MPI_COMM_WORLD);
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

void bcast_hypercube(int rank, int world_size, double* send_arr, double* recv_arr, const int arr_size)
{
    int d = floor(log2(world_size));
    int mask = int_pow(2, d) - 1, partner = 0;
    int hypercube_nodes_num = int_pow(2, d);

    //printf("%d %d\n", rank, hypercube_nodes_num);

    if (rank >= hypercube_nodes_num)
    {
        return;
    }

    int i_pow_2 = 1;

    for (int i = 0; i < d; ++i, i_pow_2 *= 2)
    {
        mask = mask ^ i_pow_2;

        if ((rank & mask) == 0)
        {
            partner = rank ^ i_pow_2;

            if ((rank & i_pow_2) == 0)
            {
                //printf("(send) %d -> %d\n", rank, partner);
                MPI_Send(send_arr, arr_size, MPI_DOUBLE, partner, SEND_PAR);
            }
            else
            {
                //printf("(recv) %d -> %d\n", partner, rank);
                MPI_Recv(recv_arr, arr_size, MPI_DOUBLE, partner, RECV_PAR);
            }
        }
    }
}

void reduce_hypercube(int rank, int world_size, double* send_arr, double* recv_arr, const int arr_size)
{
    int d = floor(log2(world_size));
    int mask = 0, partner = 0;
    int hypercube_nodes_num = int_pow(2, d);

    if (rank >= hypercube_nodes_num)
    {
        return;
    }

    int i_pow_2 = 1;

    for (int i = 0; i < d; ++i, i_pow_2 *= 2)
    {
        if ((rank & mask) == 0)
        {
            partner = rank ^ i_pow_2;

            if ((rank & i_pow_2) != 0)
            {
                //printf("(send) %d -> %d\n", rank, partner);
                MPI_Send(send_arr, arr_size, MPI_DOUBLE, partner, SEND_PAR);
            }
            else
            {
                //printf("(recv) %d -> %d\n", partner, rank);
                MPI_Recv(recv_arr, arr_size, MPI_DOUBLE, partner, RECV_PAR);

                for (int i = 0; i < arr_size; ++i)
                    send_arr[i] += recv_arr[i];
            }
        }

        mask = mask ^ i_pow_2;
    }
}


int timer_run(int rank, int world_size, const char* fname, double* send_arr, double* recv_arr, const int arr_size)
{
    double start = 0.0;
    double bcast_lin = 0.0, red_lin = 0.0, bcast_cub = 0.0, red_cub = 0.0;

    start = MPI_Wtime();
    bcast_linear(recv_arr, arr_size);
    bcast_lin = MPI_Wtime() - start;

    start = MPI_Wtime();
    reduce_linear(send_arr, recv_arr, arr_size);
    red_lin = MPI_Wtime() - start;

    start = MPI_Wtime();
    bcast_hypercube(rank, world_size, send_arr, recv_arr, arr_size);
    bcast_cub = MPI_Wtime() - start;

    start = MPI_Wtime();
    reduce_hypercube(rank, world_size, send_arr, recv_arr, arr_size);
    red_cub = MPI_Wtime() - start;

    if (rank == 0)
    {
        FILE* time_file = fopen(fname, "a+");

        if (time_file != NULL)
        {
            fprintf(time_file, "%d %lg %lg %lg %lg\n", world_size,
                    red_lin, red_cub, bcast_lin, bcast_cub);
            fclose(time_file);
            time_file = NULL;
        }
        else
        {
            printf("Failed to open file <%s> to save time measurements\n", fname);
            return 1;
        }
    }

    return 0;
}