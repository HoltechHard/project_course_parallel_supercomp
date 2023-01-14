/*
    ==================================================================================
        PARALLEL PROGRAMMING IN C - GENERIC SOLUTION FOR NEWTON-COTES INTEGRATION    |
    ==================================================================================
                                    VERSION 14.0

    Dictionary of variables: 

    k: degree of quadrature
    q: number of quadratures/ number of processes 
    n: number of partitions/intervals (defined like n = k*q)
    coeff_newton_cotes: structure of newton-cotes coefficients
        - k: position of vector of structure and represent degree of quadrature
        - Ak: coefficient for h
        - Ci: vector of weights
    part_int: partial integrals for each quadrature
    part_int{f(x), lim = [low = x0, upper = xk]} = Ak * h * sum{i = 0...k} {Ci_k * f(xi)}
    a: lower limit of definite integral
    b: upper limit of definite integral
    expression: function to integrate
    I_exact: exact value of integral

*/

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include "tinyexpr.h"
#define MAX 11
#define v 1
#define tag 100

// functions defined by programmer
void read_files(char name_file[40], char lines[5][200]);
void input_variables(double *a, double *b, int *k, char expression[200], 
                        double *I_exact, char lines[5][200]);
void print_results(double a, double b, int k, char expression[200], double I_exact);
void define_parameters(double *x, double *y, double **z);
void print_vector(double *x);
void print_matrix(double **x);
double func(double a, char expression[200]);

int main(int argc, char *argv[])
{
    // variables related with MPI
    int myid, numproc;
    // variables related with input data
    double a, b, I_exact;
    int k, q; 
    char expression[200];
    // variables related with file manager
    char name_file[40], lines[5][200];
    // variables related with newton-cotes
    double *pvec_ncotes_k, *pvec_ncotes_Ak, **pmatrix_ncotes_Ci;
    // variable to manage communication status
    MPI_Status status;
    // output variables
    double approx_int, abs_error, rel_error;

    // initialize parallel processing
    MPI_Init(&argc, &argv);
    // take the total number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &numproc);
    // take the number of the current process
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);    

    // ******************** STEP 1: broadcast input-data *************************

    // allocate memory for newton-cotes parameters
    pvec_ncotes_k = (double *) calloc(MAX, sizeof(double));
    pvec_ncotes_Ak = (double *) calloc(MAX, sizeof(double));

    pmatrix_ncotes_Ci = (double **) calloc(MAX, sizeof(double *));

    for(int i=0; i<MAX; i++)
    {
        pmatrix_ncotes_Ci[i] = (double *) calloc(MAX, sizeof(double));
    }

    // master process 0 ==> read the input data
    if(myid == 0)
    {
        printf("Name of file: ");
        
        // request the name of the file        
        scanf("%s", name_file);

        // read the txt file
        read_files(name_file, lines);

        // asign the values to input-variables
        input_variables(&a, &b, &k, expression, &I_exact, lines);
        q = numproc;

        // asign parameters of newton-cotes
        // newton-cotes parameters
        define_parameters(pvec_ncotes_k, pvec_ncotes_Ak, pmatrix_ncotes_Ci);

        // print the results of input-data readed
        if(v > 0)
        {
            printf("\n Master process 0 ");
            // print unidimensional variables
            print_results(a, b, k, expression, I_exact);
            // print newton-cotes variables
            printf("\n Newton-cotes k coefficients: \n");
            print_vector(pvec_ncotes_k);
            printf("\n Newton-cotes Ak coefficients: \n"); 
            print_vector(pvec_ncotes_Ak);
            printf("\n Newton-cotes Ci coefficients: \n");
            print_matrix(pmatrix_ncotes_Ci);            
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // broadcast all arguments from process 0 to every process [0 until numproc]
    MPI_Bcast(&a, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&b, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&q, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(expression, 200, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&I_exact, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //
    MPI_Bcast(pvec_ncotes_k, MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(pvec_ncotes_Ak, MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    for(int i=0; i<MAX; i++)
        MPI_Bcast(pmatrix_ncotes_Ci[i], MAX, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    // print the information that was obtained for each process
    if(v > 0)
    {
        printf("********** Broadcast of input-data ***********");
        printf("\n Process %d ===> variables: \n", myid);
        print_results(a, b, k, expression, I_exact);
        //
        printf("\n Newton-cotes Ak coefficients: \n"); 
        printf("\n Coeff k: \n");
        print_vector(pvec_ncotes_k);
        printf("\n Coeff Ak: \n");
        print_vector(pvec_ncotes_Ak);
        printf("\n Coeff Ci: \n");
        print_matrix(pmatrix_ncotes_Ci);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);


    // ********************* STEP 2: send/recv of original data **********************

    // variables related with computation
    int n;
    double h, *data_x, *part_data, *slice;

    // master process ==> manage data partition
    if(myid == 0)
    {
        // number of intervals
        n = k * q;
        // calculate the steps
        h = (b-a)/n;
        // allocate memory for data_x
        data_x = (double *) calloc(n+1, sizeof(double));
        
        // generate full dataset data_x
        for(int i=0; i<n+1; i++)
        {
            *(data_x + i) = a + i*h;
        }

        // print dataset
        if(v>0)
        {
            printf("\n **** All data: *** \n");
            for(int i=0; i<n+1; i++)
            {
                printf("%f \t", *(data_x + i));
            }
            printf("\n");
        }

        // PROCESS OF SENDING ...................

        // extract subsets of data for each process
        // send SLICE from process 0 ==> to ptocess i
        for(int i=1; i<q; i++)
        {
            // send h for each process
            MPI_Send(&h, 1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);
            
            // split the data in small pieces
            slice = (double *) calloc(k+1, sizeof(double));

            for(int j=0; j<k+1; j++)
            {
                *(slice + j) = *(data_x + i*k + j);
            }

            // send each slice from process 0 ==> to process i
            MPI_Send(slice, k+1, MPI_DOUBLE, i, tag, MPI_COMM_WORLD);            

            // print the send
            if(v>0)
            {
                printf("\n Process %d sent %d values to process %d  ===> \n", myid, k+1, i);
                for(int j=0; j<k+1; j++)
                    printf("%f \t", *(slice +j));
            }            
        }

        // allocate memory for part_data speciffied for process master 0
        part_data = (double *) calloc(k+1, sizeof(double));

        // fill partial-data for process 0
        for(int i=0; i<k+1; i++)
        {
            *(part_data + i) = *(data_x + i);
        }

        if(v>0)
        {
            printf("\n Process %d have this data: \n", myid);

            for(int i=0; i<k+1; i++)
                printf("%f \t", *(part_data + i));
            printf("\n");
        }

        // **************

    }else if(myid != 0){    // slaves processes ==> take the partial-data
        
        printf("\n Process %d \n", myid);

        // PROCESS OF RECEIVING..................................

        // process i <== receive h from process 0
        
        MPI_Recv(&h, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);
        // initialize part-data
        part_data = (double *) calloc(k+1, sizeof(double)); 
        // process i <== receive slice from process 0
        MPI_Recv(part_data, k+1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &status);

        if(v>0)
        {
            printf("\n Process %d <== receive %d values from process 0: \n", myid, k+1);

            for(int i=0; i<k+1; i++)
                printf("%f \t", *(part_data + i));
            printf("\n");
        }

        // **************************
    }

    // each process show the corresponding partial-data
    printf("\n ********************************* \n");
    printf("Data for process %d: \n", myid);

    for(int i=0; i<k+1; i++)
        printf("%f \t", *(part_data + i)); 
        
    printf("\n ********************************* \n");


    // ********************* STEP 3: Compute the partial integral **********************

    double *part_y, part_int, start_time, end_time, count_time;

    // initialize timing of computation calculus
    start_time = MPI_Wtime();

    // evaluate expression in vector variable ......................

    // initialize y
    part_y = (double *) calloc(k+1, sizeof(double));

    // calculate y
    for(int i=0; i<k+1; i++)
    {
        *(part_y + i) = func(*(part_data + i), expression);
    }

    // calculate partial-integral
    part_int = 0;

    for(int j=0; j<k+1; j++)
        part_int += pmatrix_ncotes_Ci[k][j] * part_y[j];

    part_int = pvec_ncotes_Ak[k] * h * part_int;
    
    // *******************************************************

    // stop clock
    end_time = MPI_Wtime();

    // count time in ms
    count_time = (end_time - start_time) * 1000;

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n +++++++++++ PARTIAL RESULTS +++++++++ \n");
    printf("\n Process %d has f(x) = \n", myid);

    for(int i=0; i<k+1; i++)
        printf("%f \t", *(part_y + i));

    printf("\n Process %d has partial-integral = %f", myid, part_int);
    printf("\n Time to calculate partial-integral: %f ms", count_time);

    MPI_Barrier(MPI_COMM_WORLD);


    // ********************* STEP 4: Calculate the approximate integral **********************

    // variables
    double final_time, *vec_pint, *vec_times, start_t2, end_t2, start_t3, end_t3;

    // master process 0 ==> manage the computation of final-results of numerical integral
    if(myid == 0)
    {
        // initialize time
        final_time = 0;

        // define array of partial-integral sums
        vec_pint = (double *) calloc(numproc, sizeof(double));

        // copy result of part-int in process 0 to partial-integral-sums[0]
        vec_pint[0] = part_int;

        // define array of time-stamps
        vec_times = (double *) calloc(numproc, sizeof(double));

        // copy the time-processing of processing 0
        vec_times[0] = count_time;

        // capture the partial-integral results and times for each process 
        // and save in vector of results in master-process
        
        for(int j=1; j<numproc; j++)
        {
            // initialize time
            start_t2 = MPI_Wtime();

            // receive the partial-integral result corresponding to j-th process
            MPI_Recv(&vec_pint[j], 1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);

            // receive the j-th speending time of process
            MPI_Recv(&vec_times[j], 1, MPI_DOUBLE, j, tag, MPI_COMM_WORLD, &status);

            // finalize time
            end_t2 = MPI_Wtime();

            // calculate total time
            final_time += (end_t2 - start_t2) * 1000;

            // print time2
            printf("\n Time to send/recv part-integrals = %f ms", (end_t2 - start_t2)*1000);

        }

        // initialize time
        start_t3 = MPI_Wtime();

        // Finnaly... calculate the approximate value of integral
        approx_int = 0;

        for(int i=0; i<numproc; i++)
        {
            approx_int += *(vec_pint + i);
        }

        // calculate absolute-error
        abs_error = fabs(I_exact - approx_int);

        // calculate relative-error
        rel_error = (abs_error/I_exact)*100;

        // finish time
        end_t3 = MPI_Wtime();

        // print time to calculate approx-integral
        printf("\n Time to calculate apprx. integral= %f ms \n", (end_t3 - start_t3)*1000);

        // calculate final accumulate time
        final_time = 0;

        for(int i=0; i<numproc; i++)
        {
            final_time += *(vec_times + i);
        }

        final_time += (end_t3 - start_t3) * 1000;

        // print the final results
        printf("\n ******************************************** \n");
        printf("FINAL RESULTS from process %d \n", myid);
        printf("\n ******************************************** \n");

        printf("\n Integral %s with limits = (%f , %f) ========> ", expression, a, b);
        printf("\n Exact-integral = %f", I_exact);
        printf("\n Approximate-integral = %.15f", approx_int);
        printf("\n Absolute-error = %.15f", abs_error);
        printf("\n Relative-error = %.15f %%", rel_error);
        printf("\n Total time to compute integral = %.8f ms", final_time);
        printf("\n ********* FINISH! ************");

    }else if(myid != 0){       // slave processes [1...q] ==> sends (part-res) ==> to master-process 0

        // send the partial-integrals-sum to master process 0
        MPI_Send(&part_int, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);

        // send the partial-integrals-sum 
        MPI_Send(&count_time, 1, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("\n");
    MPI_Finalize();

    return 0;
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

// define parameters
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



