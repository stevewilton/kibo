'''
Some Utility functions related to numpy and FixedPoint
'''

#-----------------------------------------------------------------------------
#
# Permission to use, copy, and modify this software and its documentation is
# hereby granted only under the following terms and conditions.  Both the
# above copyright notice and this permission notice must appear in all copies
# of the software, derivative works or modified versions, and any portions
# thereof, and both notices must appear in supporting documentation.
#
# This software may be distributed (but not offered for sale or transferred
# for compensation) to third parties, provided such third parties agree to
# abide by the terms and conditions of this notice.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHORS, AS
# WELL AS THE UNIVERSITY OF BRITISH COLUMBIA AND 
# THE UNIVERSITY OF SYDNEY DISCLAIM ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL THE
# AUTHORS OR THE UNIVERSITY OF BRITISH COLUMBIA OR THE 
# UNIVERSITY OF SYDNEY BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
# PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#---------------------------------------------------------------------------


import numpy as np
import random
import collections
import time
import math
from FixedPoint import *



def quantize_and_clip(arr, int_bits, frac_bits):
    max_value = (1<<(int_bits-1+frac_bits))-1
    min_value = -(1<<(int_bits-1+frac_bits))
    for i in range(0,arr.shape[0]):
       for j in range(0,arr.shape[1]):
          x = round(arr[i][j] * (1<<frac_bits))
          x = min_value if x<min_value else max_value if x > max_value else x
          arr[i][j] = float(x) / (1<<frac_bits) 

   
def return_random_np_subarray(rows, columns, fixed_point=0, int_bits=0, frac_bits=0, mean=0, stddev=0.1):
        """ return a random 2D np sub-array.  Populate it either with fixed point or floating point numbers """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if fixed_point:
                   subarray.append(FixedPoint(random.gauss(mean,stddev),int_bits,frac_bits))
                else:
                   subarray.append(random.gauss(mean,stddev))
            arr.append(subarray)
        return np.array(arr)

   
def return_one_np_subarray(rows, columns, fixed_point, int_bits, frac_bits):
        """ return a 2D np sub-array filled with 1s.  Populate it either with fixed point or floating point numbers """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if fixed_point:
                   subarray.append(FixedPoint(1.0,int_bits, frac_bits))
                else:
                   subarray.append(1.0)
            arr.append(subarray)
        return np.array(arr)

def return_zero_np_subarray(rows, columns, fixed_point, int_bits, frac_bits):
        """ return a 2D np sub-array filled with 0s.  Populate it either with fixed point or floating point numbers """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if fixed_point:
                   subarray.append(FixedPoint(0.0, int_bits, frac_bits))
                else:
                   subarray.append(0.0)
            arr.append(subarray)
        return np.array(arr)

def return_constant_np_subarray(rows, columns, val, fixed_point, int_bits, frac_bits):
        """ return a 2D np sub-array filled with a constant value.  Populate it either with fixed point or floating point numbers """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                if fixed_point:
                   subarray.append(FixedPoint(val, int_bits, frac_bits))
                else:
                   subarray.append(val)
            arr.append(subarray)
        return np.array(arr)

# Some local routines used in mydot

def convert_array_to_float(f, rows,columns):
        """ convert a fixed point np array to a floating point np array """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                   subarray.append(f[y][x].__float__() )
            arr.append(subarray)
        return np.array(arr)


def convert_array_to_fixed(f, rows,columns, int_bits, frac_bits):
        """ convert a floating point np array to a fixed point np array """
        arr = []
        for y in range(0,rows):
            subarray = []
            for x in range(0,columns):
                   subarray.append(FixedPoint(f[y][x], int_bits, frac_bits))
            arr.append(subarray)
        return np.array(arr)


def mydot(a,b):
     """ perform a dot product.  If it is fixed point, convert to floating point first, for speed """
     if isinstance(a[0][0], FixedPoint):
        int_bits = a[0][0].get_int_bits()
        frac_bits = a[0][0].get_frac_bits()
        a_float = convert_array_to_float(a, a.shape[0], a.shape[1])
        b_float = convert_array_to_float(b, b.shape[0], b.shape[1])
        c_float = np.dot(a_float, b_float)
        c_fixed = convert_array_to_fixed(c_float, c_float.shape[0], c_float.shape[1], int_bits, frac_bits)
        return(c_fixed)
     else:
        x = np.dot(a,b)
        return x

def mydot_slow(a,b):
   return(np.dot(a,b))
