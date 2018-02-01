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

f = FixedPoint(0.25,7,8)
g = FixedPoint(3.0,12,4)
f = 1/f
print f
