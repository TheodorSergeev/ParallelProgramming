/*
    Fedor Sergeev (2020)

    Build and run: make
    Alternative build: ./mainmpiCC merge_sort.cc -o main
    Alternative run: mpiexec -np 3 ./main
    (here 3 is the number of processes)
*/

#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>


#define SEND_PAR MSG_TAG, MPI_COMM_WORLD
#define RECV_PAR MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
const int MSG_TAG = 123;


int timer_run(int rank, int world_size, const char* fname);


void print_arr(int* arr, int arr_size)
{
    for (int i = 0; i < arr_size; ++i) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

int merge(int* arr1, int* arr2, int* res, int arr1_size, int arr2_size)
{
    int it_1 = 0, it_2 = 0;

    while (it_1 < arr1_size && it_2 < arr2_size) {
        if (arr1[it_1] <= arr2[it_2]) {
            res[it_1 + it_2] = arr1[it_1];
            it_1 += 1;
        } else {
            res[it_1 + it_2] = arr2[it_2];
            it_2 += 1;
        }
    }

    if (it_1 < arr1_size) {
        for (int i = it_1; i < arr1_size; ++i)
            res[it_2 + i] = arr1[i];
    } else {
        for (int j = it_2; j < arr2_size; ++j)
            res[it_1 + j] = arr2[j];
    }

    return 0;
}

int seq_merge_sort(int* arr, int arr_size)
{
    if (arr_size < 2)
        return 0;

    int half1_size = (int) floor((double) arr_size / 2);
    int half2_size = arr_size - half1_size;
    int* res = (int*) calloc(arr_size, sizeof(int));

    seq_merge_sort(arr, half1_size);
    seq_merge_sort(arr + half1_size, half2_size);
    merge(arr, arr + half1_size, res, half1_size, half2_size);

    for (int i = 0; i < arr_size; ++i) {
        arr[i] = res[i];
    }

    free(res);

    return 0;
}


void arr_decomposition(int rank, int world_size,
                       int arr_size, int* start, int* size)
{
    *start = (arr_size / world_size) * rank;
    int end = *start + (arr_size / world_size);

    if (rank == world_size - 1) {
        end = arr_size;
    }

    *size =  end - *start;
}

int lin_par_merge_sort(int* arr, int arr_size, int rank, int world_size)
{
    int start_p = -1, size_p = -1;
    arr_decomposition(rank, world_size, arr_size, &start_p, &size_p);

    // printf("rank %d  start %d  size %d\n", rank, start_p, size_p);

    seq_merge_sort(arr + start_p, size_p);

    const int ROOT_PROC_NUM = 0;

    if (rank == ROOT_PROC_NUM) {
        int start_i = -1, size_i = -1;

        for (int i = 0; i < world_size; ++i) {
            if (i == ROOT_PROC_NUM)
                continue;

            arr_decomposition(i, world_size, arr_size, &start_i, &size_i);

            // printf("%d recv from %d\n", rank, i);
            MPI_Recv(arr + start_i, size_i, MPI_INT, i, RECV_PAR);

            int* res_arr = (int*) calloc(size_p + size_i, sizeof(int));
            merge(arr + start_p, arr + start_i, res_arr, size_p, size_i);
            size_p += size_i;

            for (int j = start_p; j < size_p; ++j)
                arr[j] = res_arr[j - start_p];

            free(res_arr);
        }
        // A <- A_k

    } else {
        // printf("send from %d to %d\n", rank, ROOT_PROC_NUM);
        MPI_Send(arr + start_p, size_p, MPI_INT, ROOT_PROC_NUM, SEND_PAR);
    }
}

int int_pow(int base, int exp) {
    int result = 1;

    while (exp) {
        if (exp % 2)
           result *= base;
        exp /= 2;
        base *= base;
    }
    return result;
}

int cub_par_merge_sort(int* arr, int arr_size, int rank, int world_size)
{
    // all-to-one using hypercube topology
    int hcube_dim = floor(log2(world_size));
    int mask = 0, partner = 0;
    int hcube_nodes_num = int_pow(2, hcube_dim);

    if (rank >= hcube_nodes_num) {
        return 0;
    }

    int start_p = -1, size_p = -1;
    arr_decomposition(rank, hcube_nodes_num, arr_size, &start_p, &size_p);
    seq_merge_sort(arr + start_p, size_p);
    // printf("initial: ");
    // print_arr(arr + start_p, size_p);
    // printf("rank %d  start %d  size %d\n", rank, start_p, size_p);


    int i_pow_2 = 1;
    int start_i = -1, size_i = -1;
    MPI_Status partner_status;

    for (int i = 0; i < hcube_dim; ++i, i_pow_2 *= 2) {

        if ((rank & mask) == 0) {
            partner = rank ^ i_pow_2;

            if ((rank & i_pow_2) != 0) {
                // printf("(send) %d -> %d [%d, %d]\n", rank, partner, start_p, size_p);
                // print_arr(arr + start_p, size_p);
                MPI_Send(arr + start_p, size_p, MPI_INT, partner, SEND_PAR);
            }
            else {
                arr_decomposition(partner, hcube_nodes_num, arr_size, &start_i, &size_i);

                // the partner array size could have changed since hypercube is used
                MPI_Probe(partner, MSG_TAG, MPI_COMM_WORLD, &partner_status);
                MPI_Get_count(&partner_status, MPI_INT, &size_i);

                // printf("(recv) %d -> %d [%d, %d]\n", partner, rank, start_i, size_i);
                MPI_Recv(arr + start_i, size_i, MPI_INT, partner, RECV_PAR);

                int* res_arr = (int*) calloc(size_p + size_i, sizeof(int));

                merge(arr + start_p, arr + start_i, res_arr, size_p, size_i);
                size_p += size_i;
                start_p = 0; // quick fix

                for (int j = start_p; j < size_p; ++j)
                    arr[j] = res_arr[j - start_p];

                free(res_arr);
            }
        }

        mask = mask ^ i_pow_2;
    }

    return 0;
}

const int MAX_RANDOM_NUM = 1e6;

int random_arr_fill(int* arr, int arr_size)
{
    if (arr == NULL)
        return -1;

    for (int i = 0; i < arr_size; ++i) {
        arr[i] = rand() % MAX_RANDOM_NUM;
    }

    return 0;
}

int timer_run(int rank, int world_size, const char* fname, int arr_size)
{
    double start = 0.0;
    double t_seq = 0.0, t_par_lin = 0.0, t_par_cub = 0.0;

    int* arr = (int*) calloc(arr_size, sizeof(int));

    random_arr_fill(arr, arr_size);
    start = MPI_Wtime();
    seq_merge_sort(arr, arr_size);
    t_seq = MPI_Wtime() - start;

    random_arr_fill(arr, arr_size);
    start = MPI_Wtime();
    lin_par_merge_sort(arr, arr_size, rank, world_size);
    t_par_lin = MPI_Wtime() - start;

    random_arr_fill(arr, arr_size);
    start = MPI_Wtime();
    cub_par_merge_sort(arr, arr_size, rank, world_size);
    t_par_cub = MPI_Wtime() - start;

    free(arr);

    if(rank == 0) {
        FILE* time_file = fopen(fname, "a+");

        if (time_file != NULL) {
            fprintf(time_file, "%d %lg %lg %lg\n", world_size,
                    t_seq, t_par_lin, t_par_cub);
            fclose(time_file);
            time_file = NULL;
        } else {
            printf("Failed to open file <%s> to save time measurements\n", fname);
            return 1;
        }
    }

    return 0;
}


int main(int argc, char** argv)
{
    if (argc != 1) {
        printf("Expected the number of command line arguments to be "
               "argc=1, got %d arguments instead. Exit\n", argc);
        return 1;
    }

    int rank = 0, world_size = 0;
    const char* fname = "reduce_bcast_speed.txt";

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    const int TEST = 0;

    if (TEST == 1) {
        if (rank == 0) {
            const int arr_size = 6;

            // test merge
            int arr1[arr_size] = {1, 3, 5, 9, 10, 12};
            int arr2[arr_size] = {2, 4, 6, 8, 10, 11};
            int arr3[2 * arr_size] = {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};

            print_arr(arr1, arr_size);
            print_arr(arr2, arr_size);
            merge(arr1, arr2, arr3, arr_size, arr_size);
            print_arr(arr3, 2 * arr_size);

            // test seq_merge_sort
            int arr4[2 * arr_size] = {8, 12, 3, 5, 1, 8, 44, 1, -3, 99, 4, 6};
            print_arr(arr4, 2 * arr_size);
            seq_merge_sort(arr4, 2 * arr_size);
            print_arr(arr4, 2 * arr_size);
        }

        // test lin_par_merge_sort and cub_par_merge_sort
        const int arr5_size = 13;
        int arr5[arr5_size] = {8, 12, 3, 5, 1, 8, 44, 1, -3, 99, 4, 6, 7};

        if (rank == 0)
            print_arr(arr5, arr5_size);

        lin_par_merge_sort(arr5, arr5_size, rank, world_size);

        if (rank == 0)
            print_arr(arr5, arr5_size);

        int arr6[arr5_size] = {8, 12, 3, 5, 1, 8, 44, 1, -3, 99, 4, 6, 7};

        if (rank == 0)
            print_arr(arr6, arr5_size);

        cub_par_merge_sort(arr6, arr5_size, rank, world_size);

        if (rank == 0)
            print_arr(arr6, arr5_size);
    } else {
        const int ARRAY_SIZE = 5e6;
        timer_run(rank, world_size, "merge_sort_time.txt", ARRAY_SIZE);
    }

    MPI_Finalize();
    return 0;
}

