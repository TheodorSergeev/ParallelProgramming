/*
    Fedor Sergeev (2020) copyright

    Build: make
    Run: ./main 3
    (here 3 is the number of OpenMP threads)
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <omp.h>
#include <errno.h>
#include <cstdlib>
#include <limits.h>
#include <string.h>

#define SEND_PAR MSG_TAG, MPI_COMM_WORLD
#define RECV_PAR MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE
const int MSG_TAG = 123;


// Technical functions
void print_arr(int* arr, int arr_size);      // Print array in console
int int_pow(int base, int exp);              // Compute base in power exp for integers
const int MAX_RANDOM_NUM = 1e7;
int random_arr_fill(int* arr, int arr_size); // Fill array with random numbers <= MAX_RANDOM_NUM

// Sequential merge sort
int merge(int* arr1, int* arr2, int* res, int arr1_size, int arr2_size);
int seq_merge_sort(int* arr, int arr_size);

// Merge sort variant that uses OMP sections mechanism
int sections_par_merge_sort(int* arr, int arr_size, int threads_num);

// Merge sort variant that uses OMP tasks mechanism
const int MIN_TASK_SIZE = 100;
int tasks_par_merge_sort(int* arr, int arr_size, int min_task_size);

// Merge sort variant that uses hypercube architecture reduce
int hcube_reduce(int* arr, int arr_size, int rank, int world_size, int merge_size);
int hcube_par_merge_sort(int* arr, const int arr_size);

// Perform time trials of the merge sort variants
// ( prints merge sort variants execution time on max_thr_num threads
// for a arr_size sized array to a file named fname)
int omp_timer_run(int max_thr_num, const char* fname, int arr_size);


int main(int argc, char** argv)
{
    if (argc != 2)
    {
        printf("Expected the number of command line arguments to be "
               "argc=2, got %d arguments instead. Exit\n", argc);
        return 1;
    }

    char* argv_par_ptr = nullptr;
    errno = 0;
    long thr_num = strtol(argv[1], &argv_par_ptr, 10);

    if (errno != 0 || *argv_par_ptr != '\0' || thr_num > INT_MAX)
    {
        printf("Couldn't convert argv[1] to int, argv[1] = %s. Exit\n", argv[1]);
        return 1;
    }

    const int TEST = 0;

    if (TEST == 1)
    {
        const int arr5_size = 13;
        int arr5[arr5_size] = {8, 12, 3, 5, 1, 8, 44, 1, -3, 99, 4, 6, 7};

        print_arr(arr5, arr5_size);

        //sections_par_merge_sort(arr5, arr5_size, 3);
        //tasks_par_merge_sort(arr5, arr5_size, MIN_TASK_SIZE);
        //hcube_par_merge_sort(arr5, arr5_size);

        print_arr(arr5, arr5_size);
    }
    else
    {
        const int ARRAY_SIZE = 5e6;
        omp_timer_run(thr_num, "merge_sort_time.txt", ARRAY_SIZE);
    }

    return 0;
}


void print_arr(int* arr, int arr_size)
{
    for (int i = 0; i < arr_size; ++i)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
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

int merge(int* arr1, int* arr2, int* res, int arr1_size, int arr2_size)
{
    int it_1 = 0, it_2 = 0;

    while (it_1 < arr1_size && it_2 < arr2_size)
    {
        if (arr1[it_1] <= arr2[it_2])
        {
            res[it_1 + it_2] = arr1[it_1];
            it_1 += 1;
        }
        else
        {
            res[it_1 + it_2] = arr2[it_2];
            it_2 += 1;
        }
    }

    if (it_1 < arr1_size)
    {
        for (int i = it_1; i < arr1_size; ++i)
        {
            res[it_2 + i] = arr1[i];
        }
    }
    else
    {
        for (int j = it_2; j < arr2_size; ++j)
        {
            res[it_1 + j] = arr2[j];
        }
    }

    return 0;
}

int seq_merge_sort(int* arr, int arr_size)
{
    if (arr_size < 2)
    {
        return 0;
    }

    int half1_size = (int) floor((double) arr_size / 2);
    int half2_size = arr_size - half1_size;
    int* res = (int*) calloc(arr_size, sizeof(int));

    seq_merge_sort(arr, half1_size);
    seq_merge_sort(arr + half1_size, half2_size);
    merge(arr, arr + half1_size, res, half1_size, half2_size);

    for (int i = 0; i < arr_size; ++i)
    {
        arr[i] = res[i];
    }

    free(res);

    return 0;
}

int sections_par_merge_sort(int* arr, int arr_size, int threads_num)
{
    if (arr_size < 2)
    {
        return 0;
    }
    if (threads_num < 2)
    {
        seq_merge_sort(arr, arr_size);
    }

    #pragma omp parallel num_threads(2)
    {
        #pragma omp sections
        {
            #pragma omp section
            sections_par_merge_sort(arr,
                                    arr_size / 2,
                                    threads_num / 2);
            #pragma omp section
            sections_par_merge_sort(arr + arr_size / 2,
                                    arr_size - arr_size / 2,
                                    threads_num - threads_num / 2);
        }
    }

    int *res_arr = (int*) calloc(arr_size, sizeof(int));
    merge(arr, arr + arr_size / 2, res_arr, arr_size / 2, arr_size - arr_size / 2);
    memcpy(arr, res_arr, sizeof(int) * arr_size);
    free(res_arr);
    return 0;
}

int tasks_par_merge_sort(int* arr, int arr_size, int min_task_size)
{
    if (arr_size < 2)
    {
        return 0;
    }
    if (arr_size < min_task_size)
    {
        seq_merge_sort(arr, arr_size);
        return 0;
    }

    #pragma omp task shared(arr)
    tasks_par_merge_sort(arr,
                         arr_size / 2,
                         min_task_size);
    #pragma omp task shared(arr)
    tasks_par_merge_sort(arr + arr_size / 2,
                         arr_size- arr_size / 2,
                         min_task_size);
    # pragma omp taskwait

    int *res_arr = (int*) calloc(arr_size, sizeof(int));
    merge(arr, arr + arr_size / 2, res_arr, arr_size / 2, arr_size - arr_size / 2);
    memcpy(arr, res_arr, sizeof(int) * arr_size);
    free(res_arr);
    return 0;
}


// all-to-one using hypercube topology
int hcube_reduce(int* arr, int rank, int world_size, int merge_size)
{
    int hcube_dim = floor(log2(world_size));
    int mask = 0, partner = 0;
    int hcube_nodes_num = int_pow(2, hcube_dim);

    int i_pow_2 = 1;
    //MPI_Status partner_status;
    int new_arr_size = 0;

    for (int i = 0; i < hcube_dim; ++i, i_pow_2 *= 2)
    {
        #pragma omp barrier

        // we can't just exit the function for extra threads since we have barriers
        if (rank >= hcube_nodes_num)
        {
            continue;
        }

        partner = rank ^ i_pow_2;
        new_arr_size = merge_size * i_pow_2;

        if ((rank & mask) == 0 && (rank & i_pow_2) == 0)
        {
            int* res_arr = (int*) calloc(new_arr_size * 2, sizeof(int));

            merge (&arr[merge_size * rank], &arr[merge_size * partner], res_arr, new_arr_size, new_arr_size);
            memcpy(&arr[merge_size * rank], res_arr, sizeof(int) * new_arr_size * 2);

            free(res_arr);
            res_arr = NULL;
        }

        mask = mask ^ i_pow_2;
    }

    #pragma omp barrier

    const int ROOT = 0;
    // merge the last part to the root process
    if (rank != ROOT)
    {
        for (int i = hcube_nodes_num; i < world_size; i++)
        {
            int* res_arr = (int*) calloc(merge_size * (i + 1), sizeof(int));

            merge(arr, &arr[merge_size * i], res_arr, merge_size * i, merge_size);
            memcpy(arr, res_arr, sizeof(int) * merge_size * (i + 1));

            free(res_arr);
        }
    }

    #pragma omp barrier

    return 0;
}

int hcube_par_merge_sort(int* arr, const int arr_size)
{
    int max_thr_num = omp_get_max_threads();

    // find min el in the array
    int min_arr_el = arr[0];
    for (int i = 1; i < arr_size; ++i)
    {
        if (arr[i] < min_arr_el)
        {
            min_arr_el = arr[i];
        }
    }

    // we sort the padded array, so each proc would get arrays of equal length
    // (=> equal array merges in hcube_reduce)
    int padded_arr_size = arr_size + (max_thr_num - arr_size % max_thr_num) % max_thr_num;
    int* padded_arr = (int*) calloc(padded_arr_size, sizeof(int));
    memcpy(padded_arr, arr, arr_size * sizeof(int));

    // the array is padded with numbers that are < the min array number,
    // so we can throw the extra number after the sort
    for (int i = arr_size; i < padded_arr_size; ++i)
    {
        padded_arr[i] = min_arr_el - (i + 1 - arr_size);
    }

    assert(max_thr_num > 0);
    // average number of elements to sort per process
    int delta = padded_arr_size / max_thr_num;

    #pragma omp parallel
    {
        int rank = omp_get_thread_num();
        int size = omp_get_num_threads();

        seq_merge_sort(&padded_arr[delta * rank], delta);
        #pragma omp barrier
        hcube_reduce(padded_arr, rank, size, delta);

        #pragma omp single
        memcpy(arr, &padded_arr[(size - arr_size % size) % size], arr_size * sizeof(int));
    }

    free(padded_arr);
    padded_arr = NULL;

    return 0;
}


int random_arr_fill(int* arr, int arr_size)
{
    if (arr == NULL)
    {
        return -1;
    }

    for (int i = 0; i < arr_size; ++i)
    {
        arr[i] = rand() % MAX_RANDOM_NUM;
    }

    return 0;
}

int omp_timer_run(int thr_num, const char* fname, int arr_size)
{
    double start = 0.0;
    double t_sect = 0.0, t_task = 0.0, t_cube = 0.0;
    int* arr = (int*) calloc(arr_size, sizeof(int));
    random_arr_fill(arr, arr_size);

    omp_set_dynamic(0);           // disable dynamic teams
    omp_set_num_threads(thr_num); // set max thread number

    printf("num_threads: %d\nmax_threads: %d\n", omp_get_num_threads(),
                                                 omp_get_max_threads());

    start = omp_get_wtime();
    //omp_set_nested(1); // libgomp: Thread creation failed: Resource temporarily unavailable
    sections_par_merge_sort(arr, arr_size, thr_num);
    t_sect = omp_get_wtime() - start;

    random_arr_fill(arr, arr_size);

    int min_task_size = arr_size / thr_num;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        tasks_par_merge_sort(arr, arr_size, min_task_size);
    }
    t_task = omp_get_wtime() - start;

    random_arr_fill(arr, arr_size);
    start = omp_get_wtime();
    hcube_par_merge_sort(arr, arr_size);
    t_cube = omp_get_wtime() - start;

    printf("p = %d  t_sect = %lg  t_task = %lg  t_cube = %lg\n", thr_num, t_sect, t_task, t_cube);

    FILE* time_file = fopen(fname, "a+");

    if (time_file != NULL)
    {
        fprintf(time_file, "%d %lg %lg %lg\n", thr_num, t_sect, t_task, t_cube);
        fclose(time_file);
        time_file = NULL;
    }
    else
    {
        printf("Failed to open file <%s> to save time measurements\n", fname);
        return 1;
    }

    free(arr);
    return 0;
}
