/*
    IMPLEMENTATION OF NEWTON-COTES ALGORITHM 
    USING PTHREADS + C ==> VERSION 9.0
*/

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include "tinyexpr.h"
#define MAX 11
#define NPROC 8

// struct like global variable with data
typedef struct thread_data{
    pthread_mutex_t mutex;  // mutex for update results
    double *x_data;
    double h;
    int k;
    int q;
    char expression[200];
    int rank;
    double appx_integral;
} thr_data;

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

// functions defined by user
void generate_struct(thr_data *data, double h, int k, int q, char expression[200], double *data_x);
void * parallel_integral(void *arg);
void print_data_struct(thr_data *data);
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

    // ****** PROCESS OF SPLITTING AND DISTRIBUTED DATA ******

    // pointer for data
    thr_data *ptr_data;

    //allocate space memory for global struct with data
    ptr_data = (thr_data *) malloc(sizeof(thr_data));

    // values related with calculations
    double h, *data_x;
    int n;

    n = k * q;
    h = (b - a)/n;

    // allocate memory for data_x
    data_x = (double *) calloc(n+1, sizeof(double));

    // generate the full data set data_x
    printf("\n All data: \n");

    for(int i=0; i<n+1; i++)
    {
        *(data_x + i) = a + i*h;
        //printf("%f \t", *(data_x+i));
    }

    // fill all data in 1 unique global structure
    generate_struct(ptr_data, h, k, q, expression, data_x);

    // print data for structure
    print_data_struct(ptr_data);

    // print the global coefficients of newton-cotes
    printf("\n Print Coeff Ak: \n");
    print_vector(vec_ncotes_Ak);
    printf("\n Print Coeff Ci: \n");
    print_matrix(matrix_ncotes_Ci);

    printf("\n");

    //          ************    PARALLELIZATION        ***********

    // create variables related with managing of process
    pthread_t * lst_threads = (pthread_t *) calloc(NPROC, sizeof(pthread_t));

    // vector with positions
    int ids[NPROC];

    // initialization
    pthread_mutex_init(&ptr_data->mutex, NULL);

    // parallelize the processing for slaves

    for(int i=0; i<NPROC; i++)
    {
        ptr_data->rank = i;
        pthread_create(&lst_threads[i], NULL, parallel_integral, (void *) ptr_data);
    }

    // assign the manager of calculations to master process 0
    //ptr_data->rank = 0;
    //parallel_integral((void *) ptr_data);

    // control the callings and waiting of processes
    for(int i=0; i<NPROC; i++)
        pthread_join(lst_threads[i], NULL);

    // finalize mutex
    pthread_mutex_destroy(&ptr_data->mutex);

    // print final results
    printf("\n ********** FINAL RESULTS ********* \n");
    printf("\n Integral %s with limits = (%f , %f) ========> ", expression, a, b);
    printf("\n Exact-integral = %f", I_exact);
    printf("\n Approximate-integral = %.15f \n", ptr_data->appx_integral);
    printf("\n ********* FINISH! ************");
    printf("\n");

    free(ptr_data);
    free(lst_threads);

    return 0;
}

void generate_struct(thr_data *data, double h, int k, int q, char expression[200], double *data_x)
{
    int n;

    data->h = h;
    data->k = k; 
    data->q = q;
    n = k * q; 

    // copy vector of char
    strcpy(data->expression, expression);

    // allocate space for vector of x-data
    data->x_data = (double *) calloc(n+1, sizeof(double));
    
    // fill the values
    for(int i=0; i<n+1; i++)
    {
        data->x_data[i] = data_x[i];
    }
}

void * parallel_integral(void *arg)
{
    int rank;
    double *y, part_int;
    thr_data *data = (thr_data *) arg;
    
    // allocate space for vector y
    y = (double *) calloc(data->k+1, sizeof(double));
    rank = data->rank;

    printf("\n ==================================== \n");
    printf("\n Outputs process %d \n", rank);

    // generate the y values for each process
    for(int i=0; i<data->k+1; i++)
    {
        y[i] = func(data->x_data[rank*data->k+i], data->expression);
        printf("%f \t", y[i]);
    }

    // calculate the partial-integral
    part_int = generic_newton_cotes(y, data->h, data->k, data->q);
    printf("\n partial-integral process %d = %.15f \n", rank, part_int);

    // calculate the approximate integral
    // enter to critical region
    pthread_mutex_lock(&data->mutex);
    // update the global result
    data->appx_integral = data->appx_integral + part_int;
    // leave critical region
    pthread_mutex_unlock(&data->mutex);

    return 0;
}

void print_data_struct(thr_data *data)
{
    int n;

    printf("\n Process %d \n", data->rank);
    printf("********************* \n");
    printf("h = %f \n", data->h);
    printf("k = %d \n", data->k);
    printf("q = %d \n", data->q);
    printf("vector x: \n");

    n = data->k * data->q;

    for(int i=0; i<n+1; i++)
        printf("%f \t", data->x_data[i]);
    printf("\n");
}

// function to parallel compute partial-integrals
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




