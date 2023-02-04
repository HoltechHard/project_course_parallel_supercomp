/*
    =====================================================
    |    IMPLEMENTATION ALGORITHM NEWTON-COTES USING    |
    |            OPEN-MP + C - VERSION 2.0              |
    =====================================================

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "tinyexpr.h"
#define MAX 11

struct data{
    double *data_x;
    double h;
    int k;
    int q;
    char expression[200];
};

// auxiliary functions
void insert_struct(struct data *global_data, double *data_x, double h, int k, int q, char expression[200], int n);
void print_struct(struct data global_data, int n);
double func(double a, char expression[200]);
void read_files(char name_file[40], char lines[5][200]);
void input_variables(double *a, double *b, int *k, char expression[200], 
                        double *I_exact, char lines[5][200]);
void print_results(double a, double b, int k, char expression[200], double I_exact);
void define_parameters(double *x, double *y, double **z);
void print_vector(double *x);
void print_matrix(double **x);

int main(int argc, char *argv[])
{
    // variables related with input data
    double a, b, I_exact;
    int k, q; 
    char expression[200];
    // variables related with file manager
    char name_file[40], lines[5][200];
    // variables related with newton-cotes
    double *pvec_ncotes_k, *pvec_ncotes_Ak, **pmatrix_ncotes_Ci;

    // allocate memory for newton-cotes parameters
    pvec_ncotes_k = (double *) calloc(MAX, sizeof(double));
    pvec_ncotes_Ak = (double *) calloc(MAX, sizeof(double));

    pmatrix_ncotes_Ci = (double **) calloc(MAX, sizeof(double *));

    for(int i=0; i<MAX; i++)
    {
        pmatrix_ncotes_Ci[i] = (double *) calloc(MAX, sizeof(double));
    }

    // ********************* CONTROL OF INPUT DATA ************************

    printf("Name of file: ");
        
    // request the name of the file        
    scanf("%s", name_file);

    // read the txt file
    read_files(name_file, lines);

    // asign the values to input-variables
    input_variables(&a, &b, &k, expression, &I_exact, lines);

    #pragma omp parallel
    {
        #pragma omp single 
        {
            q = omp_get_num_threads();
            printf("\n Number of threads: %d \n", q);
        }        
    }

    // asign parameters of newton-cotes
    // newton-cotes parameters
    define_parameters(pvec_ncotes_k, pvec_ncotes_Ak, pmatrix_ncotes_Ci);

    // after read files -> print the input data
    printf("\n ***** Reading input-data ***** \n");
    // print unidimensional variables
    print_results(a, b, k, expression, I_exact);
    // print newton-cotes variables
    printf("\n Newton-cotes k coefficients: \n");
    print_vector(pvec_ncotes_k);
    printf("\n Newton-cotes Ak coefficients: \n"); 
    print_vector(pvec_ncotes_Ak);
    printf("\n Newton-cotes Ci coefficients: \n");
    print_matrix(pmatrix_ncotes_Ci);            

    //  ******************* PREPARE SHARED VARIABLES *********************

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

    // declare structure with shared data
    struct data global_data;

    // insert all input-data in global structure
    insert_struct(&global_data, data_x, h, k, q, expression, n);

    // print the data inputed in global structure
    print_struct(global_data, n);


    // ************* PROCESS PARALLEL COMPUTATION NUMERICAL APPRX-INTEGRAL **************

    // declare private variables
    double *y, *part_int, appx_int = 0.0, start_time, end_time, final_time;
    int ix, rank, j, thread_id;    

    // allocate space for vectors
    y = (double *) calloc(n+1, sizeof(double));
    part_int = (double *) calloc(global_data.q, sizeof(double));

    printf("\n Vector y: \n");

    // initialize time for parallel processing
    start_time = omp_get_wtime();

    // parallel computation of vector y
    #pragma omp parallel shared(global_data, n, y) private(ix)
    {        
        // parallel calculation of y values
        #pragma omp for
        for(ix=0; ix<n+1; ix++)
        {
            y[ix] = func(global_data.data_x[ix], global_data.expression);            
        }

        // print results
        #pragma omp for
        for(ix=0; ix<n+1; ix++)
        {
            printf("%f \t", y[ix]);
        }
    }

    // barrrier
    #pragma omp barrier

    // parallel computation of approximate integral
    #pragma omp parallel shared(global_data, pvec_ncotes_Ak, pmatrix_ncotes_Ci, y, part_int, appx_int ) \
            private(rank, j, thread_id)
    {
        // cycle to initialize vector part-int
        #pragma omp for
        for(rank=0; rank<global_data.q; rank++)
        {
            part_int[rank] = 0.0;
        }

        // cycle to compute parallel integral
        #pragma omp for
        for(rank=0; rank<global_data.q; rank++)
        {
            for(j=rank*global_data.k; j<(rank+1)*global_data.k+1; j++)
            {
                part_int[rank] += pmatrix_ncotes_Ci[global_data.k][j%global_data.k] * y[j];
            }
        }

        #pragma omp for
        for(rank=0; rank<global_data.q; rank++)
        {
            part_int[rank] = part_int[rank] * pvec_ncotes_Ak[global_data.k] * global_data.h;
        }

        // print the computation of partial-integrals
        #pragma omp for
        for(rank=0; rank<global_data.q; rank++)
        {
            thread_id = omp_get_thread_num();
            printf("\n Partial-integral from process %d ====> %.15f \n", thread_id, part_int[rank]);
        }

        #pragma omp barrier

        // make reduction => compute the total-integral
        #pragma omp for reduction(+:appx_int)
        for(rank=0; rank<global_data.q; rank++)
        {
            appx_int += part_int[rank];
        }
    }

    // ******************* FINAL RESULTS APPROX-INTEGRAL ****************

    // compute absolute and relative errors
    double abs_error, rel_error;
    abs_error = fabs(I_exact - appx_int);
    rel_error = (abs_error/I_exact)*100;

    // finalize time for processing
    end_time = omp_get_wtime();
    final_time = (end_time - start_time)*1000;

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


// ****** ********** AUXILIAR FUNCTIONS  ***************

// insert all input-data in struct
void insert_struct(struct data *global_data, double *data_x, double h, int k, int q, char expression[200], int n)
{
    // allocate memory for vector data_x
    global_data->data_x = (double *) calloc(n+1, sizeof(double));

    // fill vector data_x
    for(int i=0; i<n+1; i++)
    {
        global_data->data_x[i] = data_x[i];
    }

    global_data->h = h;
    global_data->k = k;
    global_data->q = q;
    strcpy(global_data->expression, expression);
}

// print all data which struct contains
void print_struct(struct data global_data, int n)
{
    printf("\n ********************************** \n");
    printf("\n Input data from global struct of data \n");
    printf("\n h = %f ", global_data.h);
    printf("\n k = %d ", global_data.k);
    printf("\n q = %d ", global_data.q);
    printf("\n expression = %s ", global_data.expression);

    printf("\n Vector data_x: \n");

    for(int i=0; i<n+1; i++)
    {
        printf("%f \t", global_data.data_x[i]);
    }

    printf("\n");
}

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

// define parameters of newton-cotes
void define_parameters(double *x, double *y, double **z)
{
    double vec_ncotes_k[MAX] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    
    for(int i=0; i<MAX; i++)
    {
        *(x+i) = vec_ncotes_k[i];
    }

    double vec_ncotes_Ak[MAX] = {
        0.0, 1.0/2.0, 1.0/3.0, 3.0/8.0, 2.0/45.0, 5.0/288.0,
        1.0/140.0, 7.0/17280.0, 4.0/14175.0, 9.0/89600.0, 5.0/299376.0
    };
    
    for(int i=0; i<MAX; i++)
    {
        *(y+i) = vec_ncotes_Ak[i];
    }

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

    for(int i=0; i<MAX; i++)
    {
        for(int j=0; j<MAX; j++)
        {
            z[i][j] = matrix_ncotes_Ci[i][j];
        }
    }
}

// print vectors
void print_vector(double *x)
{
    for(int i=0; i<MAX; i++)
    {
        printf("%f \t", *(x+i));
    }
}

// print matrices
void print_matrix(double **x)
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


