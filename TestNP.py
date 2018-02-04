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
 start =  time.clock()
 for i in range(0,100):
  a1 = return_random_np_subarray(n,n,0,1.0, 0,8,8)
  b1  = return_random_np_subarray(n,n,0,1.0, 0,8,8)
  c1 = mydot(a1,b1)
 end =  time.clock()
 t1 = end-start

 start =  time.clock()
 for i in range(0,100):
  a1 = return_random_np_subarray(n,n,0,1.0, 1,8,8)
  b1  = return_random_np_subarray(n,n,0,1.0, 1,8,8)
  c1 = mydot(a1,b1)
 end =  time.clock()
 t2 = end-start

 start =  time.clock()
 for i in range(0,100):
  a1 = return_random_np_subarray(n,n,0,1.0, 1,8,8)
  b1  = return_random_np_subarray(n,n,0,1.0, 1,8,8)
  c1 = mydot_slow(a1,b1)
 end =  time.clock()
 t3 = end-start

 print n,t1,t2,t3



for i in [20,40,60,80,100]:
   ex(i)
