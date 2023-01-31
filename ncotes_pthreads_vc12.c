/*
    =====================================================
        IMPLEMENTATION ALGORITHM NEWTON-COTES USING
                PTHREADS + C - VERSION 12.0
    =====================================================

*/

#define _XOPEN_SOURCE 600
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <time.h>
#include "tinyexpr.h"
#define MAX 11
#define NPROC 5

// global variables for newton-cotes coefficients
double vec_ncotes_Ak[MAX] = {
    0.0, 1.0/2.0, 1.0/3.0, 3.0/8.0, 2.0/45.0, 5.0/288.0,
    1.0/140.0, 7.0/17280.0, 4.0/14175.0, 9.0/89600.0, 5.0/299376.0
};
    
double matrix_ncotes_Ci[MAX][MAX] = {
    {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}, 
    {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {7.0, 32.0, 12.0, 32.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {19.0, 75.0, 50.0, 50.0, 75.0, 19.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    {41.0, 216.0, 27.0, 272.0, 27.0, 216.0, 41.0, 0.0, 0.0, 0.0, 0.0},
    {751.0, 3577.0, 1323.0, 2989.0, 2989.0, 1323.0, 3577.0, 751.0, 0.0, 0.0, 0.0},
    {989.0, 5888.0, -928.0, 10496.0, -4540.0, 10496.0, -928.0, 5888.0, 
                989.0, 0.0, 0.0},
    {2857.0, 15741.0, 1080.0, 19344.0, 5778.0, 5778.0, 19344.0, 1080.0,
                15741.0, 2857.0, 0.0},
    {16067.0, 106300.0, -48525.0, 272400.0, -260550.0, 427368.0, -260550.0,
                272400.0, -48525.0, 106300.0, 16067.0}
};

// define global data struct
typedef struct data{
    double *slice;
    double h;
    int k;
    int q;
    char expression[200];
    int thread_id;
} data;

pthread_mutex_t mutex;
pthread_barrier_t barrier;
double appx_int=0;

// functions to fix task
void insert_struct(data *thread_data, int rank, double h, int k, int q, char expression[200], double *data_x);
void print_struct(data thread_data);
void * parallel_integral(void *args);
double generic_newton_cotes(double *y, double h, int k, int q);

// auxiliar functions
double func(double a, char expression[200]);
void read_files(char name_file[40], char lines[5][200]);
void input_variables(double *a, double *b, int *k, char expression[200], 
                        double *I_exact, char lines[5][200]);
void print_results(double a, double b, int k, char expression[200], double I_exact);
void print_vector(double x[MAX]);
void print_matrix(double x[MAX][MAX]);


int main(int argc, char *argv[])
{
    // variables related with input data
    double a, b, I_exact;
    int k, q;
    char expression[200];
    // variables related with file manager
    char name_file[40], lines[5][200];
    // declare global data struct
    data thread_data[NPROC];


    // ******* CONTROL OF INPUT DATA *******

    printf("Name of file: ");
    
    // request the name of the file        
    scanf("%s", name_file);

    // read the txt file
    read_files(name_file, lines);

    // asign the values to input-variables
    input_variables(&a, &b, &k, expression, &I_exact, lines);
    q = NPROC;

    // check if variables went to master process
    printf("\n Reading variables from txt file \n");
    print_results(a, b, k, expression, I_exact);
    printf("\n Newton-cotes Ak coefficients: \n");
    print_vector(vec_ncotes_Ak);
    printf("\n Newton-cotes Ci coefficients: \n");
    print_matrix(matrix_ncotes_Ci);

    // ************** PROCESS OF SPLITING AND DISTRIBUTED DATA ***********

    // declare time variables
    struct timespec start_time = {0, 0}, end_time = {0, 0};
    double final_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    // declare variables to computing
    double h, *data_x;
    int n;

    n = k*q;
    h = (b - a)/n;

    // allocate memory for data_x
    data_x = (double *) calloc(n+1, sizeof(double));

    // generate full dataset
    for(int i=0; i<n+1; i++)
    {
        *(data_x + i) = a + i*h;
    }

    // generate data struct for each process
    for(int i=0; i<NPROC; i++)
        insert_struct(&thread_data[i], i, h, k, q, expression, data_x);

    // print data struct for each process
    for(int i=0; i<NPROC; i++)
        print_struct(thread_data[i]);

    // ***************** PROCESS OF COMPUTING PARTIAL-INTEGRAL *****************

    // declare threads
    pthread_t threads[NPROC];
    int rank;    

    // initialization
    pthread_mutex_init(&mutex, NULL);
    pthread_barrier_init(&barrier, NULL, NPROC);

    // parallelize integral processing for slaves
    for(rank=0; rank<NPROC;rank++)
    {
        pthread_create(&threads[rank], NULL, parallel_integral, &thread_data[rank]);
    }

    // declare array with partial integrals
    double *part_int[NPROC];

    // recover partial-integrals
    for(rank=0; rank<NPROC; rank++)
    {
        pthread_join(threads[rank], (void **) &part_int[rank]);
    }

    // finalize barrier
    pthread_barrier_destroy(&barrier);

    // finalize mutex
    pthread_mutex_destroy(&mutex);

    // ***************** FINAL RESULTS APPROX-INTEGRAL  ********************    
    
    // compute absolute and relative errors
    double abs_error, rel_error;
    abs_error = fabs(I_exact - appx_int);
    rel_error = (abs_error/I_exact)*100;

    // finish time calculation
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    final_time = (1000*(double) end_time.tv_sec + 1.0e-6*end_time.tv_nsec) - 
                    (1000*(double) start_time.tv_sec + 1.0e-6*start_time.tv_nsec);

    printf("\n ************************************************* \n");
    printf("\t \t FINAL RESULTS");
    printf("\n ************************************************* \n");
    printf("\n Integral %s with limits = (%f , %f) ========> ", expression, a, b);
    printf("\n Exact-integral = %f", I_exact);
    printf("\n Approximate-integral = %.15f", appx_int);
    printf("\n Absolute-error = %.15f", abs_error);
    printf("\n Relative-error = %.15f %%", rel_error);
    printf("\n Total time to compute integral = %.15f ms", final_time);
    printf("\n ********* FINISH! ************ \n");

    return 0;
}

//  ********** FUNCTIONS TO FIX TASK  ***********

// function to manage parallel computation of integral
void * parallel_integral(void *args)
{
    double *y, *part_int;
    data *ptr = (data *) args;

    // allocate space for variable part_int
    part_int = (double *) calloc(1, sizeof(double));

    // allocate space for vector y
    y = (double *) calloc(ptr->k+1, sizeof(double));

    printf("\n ======================================= \n");
    printf("\n Outputs for thread %d: \n", ptr->thread_id);

    // calculate the y values 
    for(int j=0; j<ptr->k+1; j++)
    {
        y[j] = func(ptr->slice[j], ptr->expression);
        printf("%f \t", y[j]);        
    }

    // wait to print all values corresponding to process
    pthread_barrier_wait(&barrier);

    // calculate the partial-integral
    *part_int = generic_newton_cotes(y, ptr->h, ptr->k, ptr->q);
    printf("\n Partial-integral for process %d = %.15f \n", ptr->thread_id, *part_int);

    // calculate the approximate integral

    // enter to critical region
    pthread_mutex_lock(&mutex);
    appx_int = appx_int + *part_int;
    // leave critical region
    pthread_mutex_unlock(&mutex);
    
    pthread_exit(part_int);

    return 0;
}

// function to compute part-integrals
double generic_newton_cotes(double *y, double h, int k, int q)
{
    double part_int = 0;

    for(int j=0; j<k+1; j++)
    {
        part_int += matrix_ncotes_Ci[k][j] * y[j];
    }

    part_int = vec_ncotes_Ak[k] * h * part_int;

    return part_int;
}

// insert input data in generic struct
void insert_struct(data *thread_data, int rank, double h, int k, int q, char expression[200], double *data_x)
{
    thread_data->thread_id = rank;
    thread_data->h = h;
    thread_data->k = k;
    thread_data->q = q;
    strcpy(thread_data->expression, expression);

    // allocate memory space for slice vector
    thread_data->slice = (double *) calloc(k+1, sizeof(double));

    // split data for each process
    for(int j=0; j<thread_data->k+1; j++)
    {
        thread_data->slice[j] = data_x[thread_data->thread_id * thread_data->k + j];
    }
}

// print data of generic struct
void print_struct(data thread_data)
{
    printf("\n ****************************** \n");
    printf("\n Thread %d ===> \n", thread_data.thread_id);
    printf("\n h = %f ", thread_data.h);
    printf("\n k = %d ", thread_data.k);
    printf("\n q = %d ", thread_data.q);
    printf("\n expression = %s ", thread_data.expression);
    
    printf("\n Input Data: \n");
    for(int i=0; i<thread_data.k+1; i++)
    {
        printf("%f \t", thread_data.slice[i]);
    }

    printf("\n");
}

//  ********** AUXILIAR FUNCTIONS  ***********

// evaluate expression in function
double func(double a, char expression[200])
{
    double x;
    int err;

    // define the input variables
    te_variable vars[] = {"x", &x};

    // compile the math-expression
    te_expr *expr = te_compile(expression, vars, 1, &err);

    if(expr)
    {
        x = a;
        const double res = te_eval(expr);
        te_free(expr);
        return res;
    }else{
        return 0;
    }
}

// define function to read files
void read_files(char name_file[40], char lines[5][200])
{
    // open file
    FILE *file = fopen(name_file, "r");
    size_t len = 200;

    // allocate memory
    char *line = malloc(sizeof(char) * len);

    // check if file to read exits
    if(file == NULL)
    {
        printf("Can't open this file ... this file not exist \n");
        return;
    }

    // save each line in array of lines
    int i=0;

    while(fgets(line, len, file)!= NULL)
    {
        strcpy(lines[i], line);
        i++;
    }

    free(line);
}

void input_variables(double *a, double *b, int *k, char expression[200], 
                        double *I_exact, char lines[5][200])
{
    *a = te_interp(lines[0], 0);
    *b = te_interp(lines[1], 0);
    *k = te_interp(lines[2], 0);
    strcpy(expression, lines[3]);
    *I_exact = te_interp(lines[4], 0);
}

void print_results(double a, double b, int k, char expression[200], double I_exact)
{
    printf("\n a = %f", a);
    printf("\n b = %f", b);
    printf("\n k = %d", k);
    printf("\n expr = %s", expression);
    printf("\n I = %f", I_exact);
}

// print vectors
void print_vector(double x[MAX])
{
    for(int i=0; i<MAX; i++)
    {
        printf("%f \t", x[i]);
    }
}

// print matrices
void print_matrix(double x[MAX][MAX])
{
    for(int i=0; i<MAX;i++)
    {
        printf("k = %d \n", i);

        for(int j=0; j<MAX; j++)
        {
            printf("%f \t", x[i][j]);
        }

        printf("\n");
    }
}

