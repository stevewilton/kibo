'''
Some Misc. functions that are useful 
'''

import numpy as np
import random
import collections
import time
import math
from FixedPoint import *
from Settings import *


   
def return_random_np_subarray(rows, columns, mean=0, stddev=0.1):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if FIXED_POINT:
                   subarray.append(FixedPoint(random.gauss(mean,stddev)))
                else:
                   subarray.append(random.gauss(mean,stddev))
            arr.append(subarray)
        return np.array(arr)

   
def return_one_np_subarray(rows, columns):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if FIXED_POINT:
                   subarray.append(FixedPoint(1.0))
                else:
                   subarray.append(1.0)
            arr.append(subarray)
        return np.array(arr)

def return_zero_np_subarray(rows, columns):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if FIXED_POINT:
                   subarray.append(FixedPoint(0.0))
                else:
                   subarray.append(0.0)
            arr.append(subarray)
        return np.array(arr)

def return_constant_np_subarray(rows, columns, val):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if FIXED_POINT:
                   subarray.append(FixedPoint(val))
                else:
                   subarray.append(val)
            arr.append(subarray)
        return np.array(arr)

def convert_array_to_float(f, rows,columns):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                   subarray.append(f[y][x].__float__() )
            arr.append(subarray)
        return np.array(arr)

def convert_array_to_fixed(f, rows,columns):
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                   subarray.append(FixedPoint(f[y][x]))
            arr.append(subarray)
        return np.array(arr)

def mydot(a,b):
     if FIXED_POINT:
        a_float = convert_array_to_float(a, a.shape[0], a.shape[1])
        b_float = convert_array_to_float(b, b.shape[0], b.shape[1]);
        c_float = np.dot(a_float, b_float)
        return(convert_array_to_fixed(c_float, c_float.shape[0], c_float.shape[1]))
     else:
        return(np.dot(a,b))

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z)) 

def tanh_prime(z): 
    return 1.0 - np.tanh(z) * np.tanh(z)

def get_cost(a, y):
    return 0.5*np.sum((a-y)*(a-y))

