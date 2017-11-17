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
from WorkloadAnomalies import *


print "Constructing data set"

training_set, validaton_set, test_set, vocab_size = WorkloadAnomalies()


print "Initializing Network"



network = Network(vocab_size)
network.add_level(0, vocab_size)
network.add_level(0, 5)
network.add_level(0, vocab_size)


print "Training Network:"

for i in range(0,len(training_set)):
   if i % 1000 == 0:
      print "training example %d/%d" % (i, len(training_set) )

   training_example_x = training_set[i][0]
   training_example_y = training_example_x
   
   o = network.train(training_example_x, training_example_y, 0.1)      


print "Trained network:"
network.print_weights_and_biases()

print "Testing network"

anomoly_correct = 0  # how many times does it correctly predict an anomoly
anomoly_wrong = 0  # how many times does it incorrectly predict an anomoly
true_correct = 0 # how many times does it correctly predict a non-anomoly
true_wrong = 0 # how manhy times does it incorrectly predict a non-anomoly

for i in range(0, len(test_set)):

   if i % 1000 == 0:
      print "test example %d/%d" % (i, len(test_set))

   test_example_x = test_set[i][0]
   test_example_y = test_set[i][1]

   o = network.forward_pass(test_example_x)
   diff = np.sum((o-test_example_x)*(o-test_example_x))

   # use an arbitary threshold of 0.5
   # why? because it works

   if (test_example_y):  # if this was supposed to be an anomoly
      if (diff > 0.5):   
          anomoly_correct += 1
      else:
          anomoly_wrong += 1
   else:   # it was not supposed to be an anomoly
      if (diff > 0.5):   
          true_wrong += 1
      else:
          true_correct += 1


print "Of the %d anomalies:" % (anomoly_correct + anomoly_wrong)
print "    correctly predicted =   %d" % anomoly_correct
print "    incorrectly predicted = %d" % anomoly_wrong
print "Of the %d non-anomalies:" % (true_correct + true_wrong)
print "    correctly predicted   = %d" % true_correct
print "    incorrectly predicted = %d" % true_wrong

 








       
