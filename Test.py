import numpy as np
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from FixedPointArray import *
from Lstm import LSTM
from Dense import Dense
from Network import *
from WorkloadChars import *
from Misc import *
from ArrayUtilities import *


a = return_random_np_subarray(100,100, 0, 1.0, 0, 0, 0)
b = return_random_np_subarray(100,100, 0, 1.0, 0, 0, 0)
f1 = FixedPointArray(a, 32, 32)
f2 = FixedPointArray(b, 32, 32)
start = time.clock()
for i in range(0,100):
   f1 = f1 + f2
end = time.clock()
print end-start
