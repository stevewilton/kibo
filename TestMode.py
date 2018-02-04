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
 a1 = return_random_np_subarray(n,n,0,1.0, 1,4,4)
 a2 = return_random_np_subarray(n,n,0,1.0, 1,4,4)
 for i in range(0,100):
  a1 = a1 * a2
 end =  time.clock()
 t1 = end-start

 print n,t1




for i in [20,40,60,80,100]:
   ex(i)
