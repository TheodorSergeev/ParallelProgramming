/*
    Fedor Sergeev (2020)

    In this version the algorithm is generilased: instead of sending/receiving
    one ghost row, here we send/receive an arbitrary number of ghost rows.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string>
#include <mpi.h>
#include <assert.h>

using std::to_string;


const int MSG_TAG = 123;
#define SEND_PAR MSG_TAG, MPI_COMM_WORLD
#define RECV_PAR MSG_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE


// Weird c-style, I know.
// This is done in half to conform to the style, that is used at seminars,
// in half to avoid using c++ with MPI (it is usually pretty ugly)
typedef struct {
    // Process parameters
    int size;           // total number of processes (world_size)
    int rank;           // id of the current process (world_rank)

    // Computational mesh parameters
    int x_mesh_size, y_mesh_size;  // mesh size along x and y
    double x_step, y_step;         // mesh step along x and y
    int t_mesh_size;               // number of time steps
    double t_step;                 // time step value

    int ghost_num;      // number of ghost cells (==number of time steps per 1 synchronisation)

    // Techical variables for the FD scheme
    double cour_x;      // courant number along x = t_step / x_step / x_step
    double cour_y;      // courant number along y= t_step / y_step / y_step

    double* sol_n;      // solution array at t = t_step * n
    double* sol_n1;     // solution array at t = t_step * (n + 1)

    // Row decomposition parameters
    int row_start;
    int row_finish;
    int row_size;   // = row_finish - row_start
} node_t;


// Computation constants
const double X_LIM = 1.0; // field size along x (width)  (x \in [0, X_LIM])
const double Y_LIM = 1.0; // field size along y (height) (y \in [0, Y_LIM])
const double T_LIM = 0.5; // computation time limit      (t \in [0, T_LIM])

enum PAR_MODE {SYNCH, ASYNCH};

int ind(node_t* node, int x, int y);
void decompose_mesh(int world_rank, int world_size, int y_size, int ghost_num,
                    int* row_start,  int* row_finish, int* row_size);
void init_mesh(node_t* node, int t_size, int x_size, int y_size, int ghost_num);
void free_mesh(node_t* node);
void initial_condition(node_t* node);
double fd_scheme(node_t* node, int x_ind, int y_ind);
void print_sol(node_t* node, const char* fname);

void seq_solver_bound_cond(node_t* node);
void seq_solver_iteration(node_t* node);
void seq_solver(node_t* node, int t_size, int x_size, int y_size);

void par_solver_bound_cond            (node_t* node);
void par_solver_iteration_synch       (node_t* node, int glob_t);
void par_solver_iteration_asynch      (node_t* node, int glob_t);
void par_solver_mult_iterations_synch (node_t* node);
void par_solver_mult_iterations_asynch(node_t* node);
double par_gather                     (node_t* node); // returns execution time

double par_solver(node_t* node, int rank, int size, int t_size,
                  int x_size, int y_size, PAR_MODE mode, int ghost_num); // returns computation time

// Run a series of time trials and record the results in a file
void timer_run(const char* fname, int t_size, int x_size, int y_size, int rank, int size);


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int size = 0, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    try
    {

        // For stability, choose the sizes so
        // t_step <= min(x_step^2 / 2, y_step^2 / 2)

        const int T_SIZE = 1e5;    // number of time steps
        const int X_SIZE = 101;    // x mesh size
        const int Y_SIZE = X_SIZE; // y mesh size

        const int TEST = 0;

        if (TEST)
        {
            PAR_MODE mode = SYNCH;

            node_t node;
            node.rank = rank;
            node.size = size;
            par_solver(&node, rank, size, T_SIZE, X_SIZE, Y_SIZE, ASYNCH, 3);
            par_gather(&node);

            if (rank == size - 1)
            {
                print_sol(&node, "solution_fin.txt");
            }
            free_mesh(&node);

            if (rank == 0 && false)
            {
                node_t seq_node;
                seq_solver(&seq_node, T_SIZE, X_SIZE, Y_SIZE);
                print_sol(&seq_node, "solution_fin.txt");
                free_mesh(&seq_node);
            }
        }
        else
        {
            const char* FNAME = "time_trials.txt";
            timer_run(FNAME, T_SIZE, X_SIZE, Y_SIZE, rank, size);
        }

    }
    catch(const char* err)
    {
        printf("ERROR: %s\n", err);
    }

    MPI_Finalize();
    return 0;
}


void timer_run(const char* fname, int t_size, int x_size, int y_size, int rank, int size)
{
    const int MIN_GHOST_NUM = 1;
    const int MAX_GHOST_NUM = 12;
    const int DELTA_GHOST_NUM = 4;
    const int ROOT = size - 1;

    node_t node;
    node.rank = rank;
    node.size = size;

    double gather_t = 0.0, synch_t = 0.0, asynch_t = 0.0;

    FILE* time_file = NULL;

    if (rank == ROOT)
    {
        time_file = fopen(fname, "a+");

        if (time_file == NULL)
        {
            printf("Failed to open file <%s> to save time measurements\n", fname);
            return;
        }
    }

    for (int ghost_num = MIN_GHOST_NUM; ghost_num <= MAX_GHOST_NUM; ghost_num += DELTA_GHOST_NUM)
    {
        synch_t  = par_solver(&node, rank, size, t_size, x_size, y_size, SYNCH, ghost_num);
        if (size > 1)
        {
            gather_t = par_gather(&node);
        }
        synch_t  += gather_t;
        printf("%d %d %lf\n", size, ghost_num, synch_t);

        if (size == 1)
            asynch_t = synch_t;
        else
            asynch_t = par_solver(&node, rank, size, t_size, x_size, y_size, ASYNCH, ghost_num);
        if (size > 1)
        {
            gather_t = par_gather(&node);
        }
        asynch_t += gather_t;

        printf("%d %d %lf\n", size, ghost_num, asynch_t);

        if (rank == ROOT)
        {
            fprintf(time_file, "%d %d %lf %lf\n", size, ghost_num, synch_t, asynch_t);
        }

        MPI_Barrier(MPI_COMM_WORLD);
    }

    if (time_file != NULL)
    {
        fclose(time_file);
        time_file = NULL;
    }
}

void par_solver_bound_cond(node_t* node)
{
    // top border
    if (node->row_start == 0)
    {
        for (int x = 0; x < node->x_mesh_size; ++x)
        {
            node->sol_n1[ind(node, x, 0)] = 0.0; //node->rank+2;//
        }
    }

    // bottom border
    if (node->row_finish == node->y_mesh_size - 1)
    {
        for (int x = 0; x < node->x_mesh_size; ++x)
        {
            node->sol_n1[ind(node, x, node->row_size - 1)] = 0.0; // node->rank+2;
        }
    }

    for (int y = 0; y < node->row_size; ++y)
    {
        node->sol_n1[ind(node, 0, y)] = 0.0;// node->rank+2; //
        node->sol_n1[ind(node, node->x_mesh_size - 1, y)] = 0.0;//node->rank+2;//
    }
}

void par_solver_iteration_synch(node_t* node, int glob_t)
{
    int left_partner  = node->rank - 1;
    int right_partner = node->rank + 1;

    if (node->rank % 2 == 1)
    {
        if (node->rank != node->size - 1)
        {
            // receive right
            MPI_Recv(node->sol_n + (node->row_size - node->ghost_num * 1) * node->x_mesh_size,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, right_partner, RECV_PAR);
            // send right
            MPI_Send(node->sol_n + (node->row_size - node->ghost_num * 2) * node->x_mesh_size,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, right_partner, SEND_PAR);
        }
        if (node->rank != 0)
        {
            // send left
            MPI_Send(node->sol_n + node->x_mesh_size * node->ghost_num,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, left_partner, SEND_PAR);
            // receive left
            MPI_Recv(node->sol_n + node->x_mesh_size * 0,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, left_partner, RECV_PAR);
        }
    }

    if (node->rank % 2 == 0)
    {
        if (node->rank != 0)
        {
            // send left
            MPI_Send(node->sol_n + node->x_mesh_size * node->ghost_num,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, left_partner, SEND_PAR);
            // receive left
            MPI_Recv(node->sol_n + node->x_mesh_size * 0,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, left_partner, RECV_PAR);
        }
        if (node->rank != node->size - 1)
        {
            // receive right
            MPI_Recv(node->sol_n + (node->row_size - node->ghost_num * 1) * node->x_mesh_size,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, right_partner, RECV_PAR);
            // send right
            MPI_Send(node->sol_n + (node->row_size - node->ghost_num * 2) * node->x_mesh_size,
                     node->x_mesh_size * node->ghost_num,
                     MPI_DOUBLE, right_partner, SEND_PAR);
        }
    }

    for (int t = 1; t <= node->ghost_num; ++t)
    {
        par_solver_bound_cond(node);

        if (glob_t + t >= node->t_mesh_size)
        {
            break;
        }

        for (int i = 1; i < node->x_mesh_size - 1; ++i)
        {
            for (int j = t; j < node->row_size - t; ++j)
            {
                node->sol_n1[ind(node, i, j)] = fd_scheme(node, i, j);//node->rank+1;//
            }
        }

        std::swap(node->sol_n, node->sol_n1);
    }
}

void par_solver_iteration_asynch(node_t* node, int glob_t)
{
    int left_partner  = node->rank - 1;
    int right_partner = node->rank + 1;

    MPI_Request left_request;
    MPI_Request right_request;

    // asynchronous send right
    if (node->rank != node->size - 1)
        MPI_Isend(node->sol_n + (node->row_size - node->ghost_num * 2) * node->x_mesh_size,
                  node->x_mesh_size * node->ghost_num,
                  MPI_DOUBLE, right_partner, SEND_PAR, &left_request);

    // asynchronous receive left
    if (node->rank != 0)
        MPI_Irecv(node->sol_n + node->x_mesh_size * 0,
                  node->x_mesh_size * node->ghost_num,
                  MPI_DOUBLE, left_partner, MSG_TAG, MPI_COMM_WORLD, &left_request);

    // asynchronous send left
    if (node->rank != 0)
        MPI_Isend(node->sol_n + node->x_mesh_size * node->ghost_num,
                  node->x_mesh_size * node->ghost_num,
                  MPI_DOUBLE, left_partner, SEND_PAR, &right_request);

    // asynchronous receive right
    if (node->rank != node->size - 1)
        MPI_Irecv(node->sol_n + (node->row_size - node->ghost_num * 1) * node->x_mesh_size,
                  node->x_mesh_size * node->ghost_num,
                  MPI_DOUBLE, right_partner, MSG_TAG, MPI_COMM_WORLD, &right_request);

    MPI_Wait(&left_request,  MPI_STATUS_IGNORE);
    MPI_Wait(&right_request, MPI_STATUS_IGNORE);

    int left_border  = node->rank == 0            ? 0                  : node->ghost_num;
    int right_border = node->rank == node->size-1 ? node->row_size - 1 : node->row_size - node->ghost_num - 1 ;

    //printf("%d %d %d\n", node->rank, left_border, right_border);

    // copy sol_n to sol_n1 --- this preserves the border cells for
    // when swapping sol_n1 and sol_n when computing center cells
    for (int i = 1; i <= node->x_mesh_size - 1; ++i)
    {
        for (int j = 0; j < node->row_size; ++j)
        {
            node->sol_n1[ind(node, i, j)] = node->sol_n[ind(node, i, j)];
        }
    }

    // compute center cells
    for (int t = 1; t <= node->ghost_num; ++t)
    {
        par_solver_bound_cond(node);

        if (glob_t + t >= node->t_mesh_size)
        {
            break;
        }

        for (int i = 1; i < node->x_mesh_size - 1; ++i)
        {
            for (int j = left_border + t; j < right_border - t; ++j)
            {
                node->sol_n1[ind(node, i, j)] = fd_scheme(node, i, j);//node->rank+1;//
            }
        }

        std::swap(node->sol_n, node->sol_n1);
    }

    // copy sol_n to sol_n1 --- this preserves the center cells for
    // when swapping sol_n1 and sol_n when computing border cells
    for (int i = 1; i <= node->x_mesh_size - 1; ++i)
    {
        for (int j = left_border; j < right_border; ++j)
        {
            node->sol_n1[ind(node, i, j)] = node->sol_n[ind(node, i, j)];
        }
    }

    // compute border cells
    for (int t = 1; t <= node->ghost_num; ++t)
    {
        par_solver_bound_cond(node);

        if (glob_t + t >= node->t_mesh_size)
        {
            break;
        }

        for (int i = 1; i < node->x_mesh_size - 1; ++i)
        {

            if (node->rank != 0)
            {
                for (int j = t; j <= t + node->ghost_num; ++j)
                {
                    node->sol_n1[ind(node, i, j)] = fd_scheme(node, i, j);//node->rank+1.5;//
                }
            }

            if (node->rank != node->size - 1)
            {
                for (int j = node->row_size - t - 1; j >= node->row_size - t - 1 - node->ghost_num; --j)
                {
                    node->sol_n1[ind(node, i, j)] = fd_scheme(node, i, j);//node->rank+1;//
                }
            }
        }

        std::swap(node->sol_n, node->sol_n1);
    }
}

double par_solver(node_t* node, int rank, int size, int t_size, int x_size, int y_size, PAR_MODE mode, int ghost_num)
{
    node->rank = rank;
    node->size = size;

    init_mesh(node, t_size, x_size, y_size, ghost_num);
    initial_condition(node);

    double time = MPI_Wtime();

    for (int t = 0; t < node->t_mesh_size; t += node->ghost_num)
    {

        switch (mode)
        {
            case SYNCH:
                par_solver_iteration_synch(node, t);
                break;

            case ASYNCH:
                par_solver_iteration_asynch(node, t);
                break;
        }
    }

    return MPI_Wtime() - time;
}


double par_gather(node_t* node)
{
    double time = MPI_Wtime();

    double* recv_buf = nullptr;
    const int root = node->size - 1;

    if (node->rank == root)
    {
        recv_buf = (double*) calloc(node->x_mesh_size * node->y_mesh_size, sizeof(double));
    }

    // prepare Gatherv parameters - receive counts and displacement arrays
    int recvcounts[node->size];
    int displs    [node->size];
    int row_start  = 0;
    int row_finish = 0;
    int row_size   = 0;

    int delta = (int) floor(node->y_mesh_size / node->size); // number of rows for one process on average

    for (int rank = 0; rank < node->size; ++rank)
    {
        recvcounts[rank] = node->x_mesh_size * delta;
        displs    [rank] = node->x_mesh_size * delta * rank;

        //if (rank == root) printf("rank %d  recv count %d  displs %d\n", rank, recvcounts[rank], displs[rank]);
    }

    int left_tab = node->x_mesh_size * node->ghost_num;
    if (node->rank == 0)
    {
        left_tab = 0;
    }
    if (node->rank == node->size - 1)
    {
        recvcounts[node->rank] = node->x_mesh_size * (node->y_mesh_size - delta+1);

        //if (node->rank == root) printf("rank %d  recv count %d  displs %d\n", node->rank, recvcounts[node->rank], displs[node->rank]);
    }

    MPI_Gatherv(node->sol_n + left_tab, node->x_mesh_size * delta, MPI_DOUBLE,
                recv_buf, recvcounts, displs, MPI_DOUBLE,
                root, MPI_COMM_WORLD);

    if (node->rank == root)
    {
        free(node->sol_n);
        node->sol_n = recv_buf;
    }

    node->row_start = 0;
    node->row_finish = node->y_mesh_size;
    node->row_size = node->y_mesh_size;

    return MPI_Wtime() - time;
}


void decompose_mesh(int world_rank, int world_size, int y_size, int ghost_num,
                    int* row_start,  int* row_finish, int* row_size)
{
    // number of rows for one process on average
    int delta = (int) floor(y_size / world_size);

    *row_start  = delta * world_rank - ghost_num;
    *row_finish = delta * world_rank + delta + ghost_num;

    if (world_rank == 0)
    {
        *row_start = 0;
    }

    if (world_rank == world_size - 1)
    {
        *row_finish = y_size - 1;
    }

    // deal with situation, when number of processes is > mesh size
    // in this case we ignore ghost_num
    /*if (node->size >= y_size)
    {
        node->row_start  = std::min(node->rank, x_size);
        node->row_finish = std::min(node->rank + 1, x_size);
        node->ghost_num  = 1;
    }*/

    *row_size = *row_finish - *row_start + 1;
}

void init_mesh(node_t* node, int t_size, int x_size, int y_size, int ghost_num)
{
    if (node == nullptr)
    {
        throw "node pointer is null";
    }
    if (t_size <= 0)
    {
        throw "t_size <= 0";
    }
    if (x_size <= 0 || y_size <= 0)
    {
        throw "x_size or y_size <= 0";
    }
    if (ghost_num < 1 || ghost_num >= x_size || ghost_num >= y_size)
    {
        throw "invalid ghost num";
    }

    node->x_mesh_size = x_size;
    node->y_mesh_size = y_size;
    node->x_step = X_LIM / (x_size - 1);
    node->y_step = Y_LIM / (y_size - 1);

    node->t_mesh_size = t_size;
    node->t_step = T_LIM / t_size;

    node->cour_x = node->t_step / (node->x_step * node->x_step);
    node->cour_y = node->t_step / (node->y_step * node->y_step);

    node->ghost_num = ghost_num;

    decompose_mesh(node->rank, node->size, y_size, ghost_num,
                   &node->row_start, &node->row_finish, &node->row_size);

    // printf("rank = %2d  rows = [%2d, %2d] %d\n", node->rank, node->row_start, node->row_finish, node->row_size);

    node->sol_n  = (double*) calloc(x_size * node->row_size, sizeof(double));
    node->sol_n1 = (double*) calloc(x_size * node->row_size, sizeof(double));

    if (node->sol_n == nullptr || node->sol_n1 == nullptr)
    {
        throw "Couldn't allocate meshes (size = " + to_string(x_size * node->row_size) + ")";
    }
}

void free_mesh(node_t* node)
{
    if (node->sol_n != nullptr)
    {
        free(node->sol_n);
    }
    if (node->sol_n1 != nullptr)
    {
        free(node->sol_n1);
    }

    node->sol_n  = nullptr;
    node->sol_n1 = nullptr;
}

/*  Note that this is not an effective implementation of indexing
 *  (extra multiplication for every indexing operation)
 *  However, it simplifies the code.
 *  This task is a practice task, so I will leave it as is
 */
int ind(node_t* node, int x, int y)
{
    return node->x_mesh_size * y + x;
}

void initial_condition(node_t* node)
{
    double x = 0.0;
    double y = 0.0;

    const double INIT_X_LEFT  = 0.4 / node->x_step;
    const double INIT_Y_LEFT  = 0.4 / node->y_step;
    const double INIT_X_RIGHT = 0.6 / node->x_step;
    const double INIT_Y_RIGHT = 0.6 / node->y_step;

    for (int i = 0; i < node->x_mesh_size; ++i)
    {
        for (int j = 0; j < node->row_size; ++j)
        {
            if (i >= INIT_X_LEFT && i <= INIT_X_RIGHT &&
                j + node->row_start >= INIT_Y_LEFT && j + node->row_start <= INIT_Y_RIGHT)
            {
                node->sol_n[ind(node,i,j)] = 1.0;
            }
            else
            {
                node->sol_n[ind(node,i,j)] = 0.0;
            }

        }
    }
}

double fd_scheme(node_t* node, int x_ind, int y_ind)
{
    assert(x_ind > 0 && x_ind < node->x_mesh_size - 1);
    assert(y_ind > 0 && y_ind < node->row_size - 1);

    int i = x_ind;
    int j = y_ind;

    return node->sol_n[ind(node, i, j)] +
      node->cour_x * (node->sol_n[ind(node, i - 1, j)] - 2 * node->sol_n[ind(node, i, j)] + node->sol_n[ind(node, i + 1, j)]) +
      node->cour_y * (node->sol_n[ind(node, i, j - 1)] - 2 * node->sol_n[ind(node, i, j)] + node->sol_n[ind(node, i, j + 1)]);
}

void print_sol(node_t* node, const char* fname)
{
    void* res = freopen(fname, "w", stdout);

    if (res == nullptr)
    {
        throw "Couldn't open file " + std::string(fname) + " for saving.";
    }
    if (node == nullptr || node->sol_n == nullptr)
    {
        throw "solution array is NULL";
    }

    for (int i = 0; i < node->x_mesh_size; ++i)
    {
        for (int j = 0; j < node->row_size; ++j)
        {
            printf("%lf", node->sol_n[ind(node, i, j)]);

            if (j != node->row_size - 1)
            {
                printf(",");
            }
        }
        printf("\n");
    }

    fclose(stdout);
}



void seq_solver_bound_cond(node_t* node)
{
    for (int x = 0; x < node->x_mesh_size; ++x)
    {
        node->sol_n1[ind(node, x, 0)] = 0.0;
        node->sol_n1[ind(node, x, node->y_mesh_size - 1)] = 0.0;
    }
    for (int y = 0; y < node->y_mesh_size; ++y)
    {
        node->sol_n1[ind(node, 0, y)] = 0.0;
        node->sol_n1[ind(node, node->x_mesh_size - 1, y)] = 0.0;
    }
}

void seq_solver_iteration(node_t* node)
{
    for (int i = 1; i < node->x_mesh_size - 1; ++i)
    {
        for (int j = 1; j < node->y_mesh_size - 1; ++j)
        {
            node->sol_n1[ind(node, i, j)] = fd_scheme(node, i, j);
        }
    }
}

void seq_solver(node_t* node, int t_size, int x_size, int y_size)
{
    node->rank = 0;
    node->size = 1;
    init_mesh(node, t_size, x_size, y_size, 1);
    initial_condition(node);

    for (int t = 0; t < 10; ++t) //node->t_mesh_size
    {
        seq_solver_bound_cond(node);
        seq_solver_iteration(node);
        std::swap(node->sol_n, node->sol_n1);
    }
}
