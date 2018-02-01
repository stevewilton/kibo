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



print "doing floating point test"
start =  time.clock()
for i in range(0,100):
  a1 = return_random_np_subarray(100,100,0,1.0, 0,8,8)
  b1  = return_random_np_subarray(100,100,0,1.0, 0,8,8)
  c1 = mydot(a1,b1)
end =  time.clock()
print "time = ",end-start

print "doing fixed point test"
start =  time.clock()
for i in range(0,100):
  a1 = return_random_np_subarray(100,100,0,1.0, 1,8,8)
  b1  = return_random_np_subarray(100,100,0,1.0, 1,8,8)
  c1 = mydot(a1,b1)
end =  time.clock()
print "time = ",end-start
