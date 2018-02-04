import numpy as np
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from Lstm import LSTM
from Dense import Dense
from Network import *
from WorkloadChars import *
from Misc import *
from ArrayUtilities import *


def ex(n):

 t1 = []
 for rep in [32]:
    start =  time.clock()
    a1 = return_random_np_subarray(n,n,0,1.0, 1,rep,rep)
    a2 = return_random_np_subarray(n,n,0,1.0, 1,rep,rep)
    for i in range(0,100):
       a1 = a1 + a2

    end =  time.clock()
    t1.append(end-start)
 print n,t1



for i in [100]:
   ex(i)


a = FixedPoint(2,8,4)
b = FixedPoint(3,8,4)
c = a + b
print a,b,c

print "_________"

a = np.array( [[1.1,2.2,3.3],[4.4,5.5,6.6]] )

print a
quantize_and_clip(a, 8, 2)
print a
