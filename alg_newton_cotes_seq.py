#######################################################
#        GENERIC ALGORITHM OF NEWTON-COTES            #
#######################################################

# dictionary of variables
# k: degree of quadrature
# q: number of quadratures (#iterations of the same quadrature)
# n: number of intervals/partitions (defined like n = k*q)
# coeff_newton_cotes: table of newton-cotes coefficient
#   - Ak: coefficient of h
#   - Ci: vector of weights 
# partial integrals of each quadrature:
# I_part{f(x), lim = [inf = x0, sup = xk]} = Ak * h * sum{i = 0..k} {Ci_k * f(xi)}
# a: lower limit of definite integral
# b: upper limit of definite integral
# expresion: function to integrate
# I_exact: exact value of integral

import numpy as np
import math

################################
#        Read Input data       #
################################

# read the content of the file
file_name = str(input("Name of the file [folder/file.txt]: "))

with open(file_name) as file:
    lines = [line.strip() for line in file]

# define input variables
a = float(eval(lines[0]))
b = float(eval(lines[1]))
k = int(eval(lines[2]))
q = int(eval(lines[3]))
expression = lines[4]
I_exact = float(eval(lines[5]))

# interpreter for math expressions
def f(x):
    f = eval(expression)
    return f

# positions
# i: [0, 1, 2, ..., n-1, n] ==> #intervals = n || #total of points = n + 1

# generic method for execute numeric integration algorithms
def generic_integral_rule(a, b, k, q, I_exact):
    #number of intervals
    n = k * q
    # compute step
    h = (b - a)/n    
    # generate X
    data_x = np.linspace(start = a, stop = b, num = n + 1)
    # initialize Y
    data_y = np.zeros(n + 1)
    # generate Y
    for i in range(n + 1):
        data_y[i] = f(data_x[i])
    # calculate aproximate integral 
    I_aprox = generic_newtonCotes(data_y, h, k, q)
    # calculate error
    error = I_exact - I_aprox

    return I_aprox, error

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

# compute the apx-integral using sum of partial-integrals with rule newton-cotes
def generic_newtonCotes(y, h, k, q):
    apx_int = 0 
    # i: index of current quadrature
    i = 1
    while i <= q:
        partial_int = 0
        for j in range((i-1)*k, i*k + 1):
            partial_int += newton_cotes_coeff[k]['Ci'][j%k] * y[j]
        apx_int += partial_int
        i += 1
    apx_int = newton_cotes_coeff[k]['Ak'] * h * apx_int

    return apx_int

# print result of approximate integral and their error
res_aprox, error_aprox = generic_integral_rule(a, b, k, q, I_exact)
print("Integral of ", expression)
print("Exact-integral = ", I_exact)
print("Aproximate-integral = ", res_aprox, " ........ Error = ", error_aprox)

