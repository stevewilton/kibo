'''
Run file
'''

import numpy as np
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from Lstm import LSTM
from Dense import Dense
from Settings import *
from Network import *
from WorkloadFMAnomalies import *


# Hyper-parameters
FFT_WINDOW_SIZE = 32
LAYER_SIZES = [16,8,16,32]

print "Constructing data set"

training_set, validation_set, test_set = WorkloadFMAnomalies(FFT_WINDOW_SIZE)


print "Initializing Network"

network = Network(FFT_WINDOW_SIZE)
for layer_size in LAYER_SIZES:
    network.add_level(0,layer_size)

print "Training Network:"

final_err = 0
err = 0
for i in range(0,len(training_set)):

   training_example_x = training_set[i][0]
   training_example_y = training_example_x

   err += network.train(training_example_x, training_example_y, 0.1)

   final_err = err
   if i % 100 == 0:
      print "training example %d/%d,  err=%f" % (i, len(training_set), err/100 )
      err = 0

   
print "Trained network:"
network.print_weights_and_biases()

print "Testing network"

anomaly_correct = 0  # how many times does it correctly predict an anomaly
anomaly_wrong = 0  # how many times does it incorrectly predict an anomaly
true_correct = 0 # how many times does it correctly predict a non-anomaly
true_wrong = 0 # how manhy times does it incorrectly predict a non-anomaly

for i in range(0, len(test_set)):

   if i % 10 == 0:
      print "test example %d/%d" % (i, len(test_set))

   test_example_x = test_set[i][0]
   test_example_y = test_set[i][1]

   o = network.forward_pass(test_example_x)
   diff_r = np.sum((o-test_example_x)*(o-test_example_x))

   if FIXED_POINT:
      diff = diff_r.val()
   else:
      diff = diff_r     


   # use an arbitary threshold of 0.5
   # why? because it works

   if (test_example_y):  # if this was supposed to be an anomaly
      if (diff > final_err*1.1):
          anomaly_correct += 1
      else:
          anomaly_wrong += 1
   else:   # it was not supposed to be an anomaly
      if (diff > final_err*1.1):
          print diff
          true_wrong += 1
      else:
          true_correct += 1


print "Of the %d anomalies:" % (anomaly_correct + anomaly_wrong)
print "    correctly predicted =   %d" % anomaly_correct
print "    incorrectly predicted = %d" % anomaly_wrong
print "Of the %d non-anomalies:" % (true_correct + true_wrong)
print "    correctly predicted   = %d" % true_correct
print "    incorrectly predicted = %d" % true_wrong

