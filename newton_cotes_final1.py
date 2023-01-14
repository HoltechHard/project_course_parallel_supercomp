##########################################################################################
#      PARALLEL PROGRAMMING - GENERIC-NUMERICAL SOLUTION FOR NEWTON-COTES INTEGRATION    #
##########################################################################################

# dictionary of variables
# k: degree of quadrature
# q : number of quadratures/ number of processes
# n: number of intervals/partitions (defined like n = k*q)
# coeff_newton_cotes: table of newton-cotes coefficient
#   - Ak: coefficient of h
#   - Ci: vector of weights 
# partial integrals of each quadrature:
# I_part{f(x), lim = [low = x0, upper = xk]} = Ak * h * sum{i = 0..k} {Ci_k * f(xi)}
# a: lower limit of definite integral
# b: upper limit of definite integral
# expresion: function to integrate
# I_exact: exact value of integral

# import packages
from mpi4py import MPI
import numpy as np
import math

# generic parameters 
comm = MPI.COMM_WORLD
numproc = comm.Get_size()
myid = comm.Get_rank()

# function to read files
def read_file():
    # read line for line the text file    
    file_name = str(input("Name of file: "))
    with open(file_name) as file:
        lines = [line.strip() for line in file]
    # define input variables
    a = float(eval(lines[0]))
    b = float(eval(lines[1]))
    k = int(eval(lines[2]))    
    expression = lines[3]
    I_exact = float(eval(lines[4]))
    return a, b, k, expression, I_exact    

#########################################
#    MAIN: process of paralelization    #
#########################################

##### paralelization process 1: broadcast of input-data #####

comm.Barrier()

# master process 0
if(myid == 0):
    # get the variables
    print("Name of file: ")
    a, b, k, expression, I_exact = read_file()
    q = numproc
    
    # Dictionary of Newton-Cotes coefficients
    #link: https://mathworld.wolfram.com/Newton-CotesFormulas.html

    # quadratures: k = 1...10 (from linear functions to power 10)
    newton_cotes_coeff = {
        1: {'Ak': 1/2, 'Ci': [1, 1]},
        2: {'Ak': 1/3, 'Ci': [1, 4, 1]},
        3: {'Ak': 3/8, 'Ci': [1, 3, 3, 1]},
        4: {'Ak': 2/45, 'Ci': [7, 32, 12, 32, 7]},
        5: {'Ak': 5/288, 'Ci': [19, 75, 50, 50, 75, 19]},
        6: {'Ak': 1/140, 'Ci': [41, 216, 27, 272, 27, 216, 41]},
        7: {'Ak': 7/17280, 'Ci': [751, 3577, 1323, 2989, 2989, 1323, 3577, 751]},
        8: {'Ak': 4/14175, 'Ci': [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989]},
        9: {'Ak': 9/89600, 'Ci': [2857, 15741, 1080, 19344, 5778, 5778, 19344, 1080, 15741, 2857]},
        10: {'Ak': 5/299376, 'Ci': [16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, 
                                    -48525, 106300, 16067]}
    }

elif(myid != 0):    # slave processes
    # initialization of variables
    a = b = k = q = expression = I_exact =  None
    newton_cotes_coeff = {}
    print("This is the process ", myid)

comm.Barrier()

# broadcast all arguments from process 0 to every process [1, ... numproc]
a = comm.bcast(a, root = 0)
b = comm.bcast(b, root = 0)
k = comm.bcast(k, root = 0)
q = comm.bcast(q, root = 0)
expression = comm.bcast(expression, root = 0)
I_exact = comm.bcast(I_exact, root = 0)
newton_cotes_coeff = comm.bcast(newton_cotes_coeff, root = 0)

# print the information that was obtained for each process
print(" ############# ")
print("Process ", myid, " ===> variables: ")
print("a = ", a)
print("b = ", b)
print("k = ", k)
print("q = ", q)
print("expression = ", expression)
print("I = ", I_exact)
print("Newton-cotes coeff = ", newton_cotes_coeff)

comm.Barrier()

###### paralelization process 2: send/recv of original data #######

if(myid == 0):
    # number of intervals
    n = k * q
    # compute step
    h = (b - a) / n
    # generate X
    data_x = np.linspace(start = a, stop = b, num = n + 1)

    # extract subsets of data for each process
    # send SLICE from process 0 ===> to process i
    for i in range(1, q):
        # send h for each process
        comm.send(h, dest = i)        
        # split the data
        slice = data_x[i*k:(i+1)*k+1]
        # print the send
        print("Process ", myid, " sends ", (k+1), " values to process ", i, " ===> ", slice)
        # send each slice from process 0 ===> to process i
        comm.send(slice, dest = i)
    # partial-data from process 0
    part_data = data_x[0:(k+1)]
    print("Process ", myid, " have this data: ", part_data)
else:
    # initialize h
    h = None
    # process i <== receive h from process 0
    h = comm.recv(source = 0)
    # initialize partial-data
    part_data = np.empty(k + 1, dtype = np.float64)
    # process i <=== receive slice from process 0
    part_data = comm.recv(source = 0)
    # print the receiver
    print("Process ", myid, " <=== receive ", (k+1), " values from process 0: ", part_data)

comm.Barrier()

# each process show the corresponding partial-data
print(myid, " process ======> ", part_data)

comm.Barrier()

############################################
#       Compute the partial-integral       #
############################################

# internal interpreter for string-math expressions
def f(x):
    f = eval(expression)
    return f

# start clock
start_time = MPI.Wtime()

# initialize y
part_y = np.zeros(k + 1)

# calculate y
for i in range(k + 1):
    part_y[i] = f(part_data[i])

# calculate part-int
part_int = 0

for j in range(k + 1):
    part_int += newton_cotes_coeff[k]['Ci'][j] * part_y[j]
part_int = newton_cotes_coeff[k]['Ak'] * h * part_int

# stop clock
end_time = MPI.Wtime()

# count time
count_time = (end_time - start_time) * 1000

comm.Barrier()

print("Process ", myid, " has f(x) ===> ", part_y, " with partial-integral = ", part_int)
print("Time to calculate partial-integral = ", count_time, " ms")

comm.Barrier()

##################################################
# FINAL STEP: CALCULATE THE APROXIMATE INTEGRAL  #
##################################################

# master-process 0 ==> manage the computation of final-results of integral
if myid == 0:    
    final_time = 0
    # define array of partial integral-sums and
    # initialize array with [0, ... 0] 
    vec_pint = np.zeros(numproc, np.float64)
    # copy result of part-int in process 0 to vector of partial-integral-sums[0]
    vec_pint[0] = part_int
    # define array of time-stamp
    vec_times = np.zeros(numproc, np.float64)
    # copy time-processing of process 0
    vec_times[0] = count_time
    # capture partial-integral results and times for each process 
    # and save in vector of results in master-process
    for j in range(1, numproc):
        start_t2 = MPI.Wtime()
        vec_pint[j] = comm.recv(source = j)
        vec_times[j] = comm.recv(source = j)
        end_t2 = MPI.Wtime()
        final_time += (end_t2 - start_t2) * 1000
        print("Time to send/recv part-integrals = ", (end_t2 - start_t2) * 1000, " ms")
    start_t3 = MPI.Wtime()
    # finnaly, calculate the value of approximate-integral
    aprox_int = sum(vec_pint)
    # calculate absolute-error
    abs_error = abs(I_exact - aprox_int)
    # calculate relative-error
    rel_error = (abs_error / I_exact) * 100
    end_t3 = MPI.Wtime()
    # calculate final time
    final_time += sum(vec_times)
    final_time += (end_t3 - start_t3) * 1000
    print("Time to calculate approx-integral = ", (end_t3 - start_t3) * 1000, " ms")
    # PRINT THE FINAL RESULTS
    print("***** FINAL RESULTS ***** from process ", myid)
    print("Integral ", expression, " with limits = ( ", a, ", ", b, ") ===> ")
    print("Exact-integral = ", I_exact)
    print("Approximate-integral = ", aprox_int)
    print("Absolute-error = ", abs_error)
    print("Relative-error = ", rel_error, "%")
    print("Total-time to compute aproximate integral = ", final_time, " ms")
    print("Finish!")
else:   # slave processes [1..q] ==> sends part_int (partial-results) ===> to master-process 0
    comm.send(part_int, dest = 0)
    comm.send(count_time, dest = 0)

comm.Barrier()

