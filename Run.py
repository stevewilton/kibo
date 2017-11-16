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
from WorkloadChars import *


print "Constructing data set"

training_set, validaton_set, test_set, vocab_size, reverse_dictionary = WorkloadChars()

print "Initializing Network"

network = Network(vocab_size, vocab_size, vocab_size, 0)




outstr = ""

print "Training Network"

for i in range(0,len(training_set)):
   if i % 100 == 0:
      print "training example %d/%d" % (i, len(training_set) )

   training_example_x = training_set[i][0]
   training_example_y = training_set[i][1]

   for l in range(0, len(training_example_x) ):
      o = network.train(training_example_x[l], training_example_y[l], 0.1)      

print "Trained network:"
network.print_weights_and_biases()

print "Testing network"

for i in range(0, len(test_set)):
   if i % 1000 == 0:
      print "test example %d/%d" % (i, len(test_set))

   test_example_x = test_set[i][0]
   test_example_y = test_set[i][1]
   
   last_output = " "   
   outstr = "Seeding with: >"
   for l in range(0, 10):
       last_output, last_h = network.forward_pass(test_example_x[l])
       outstr = outstr + reverse_dictionary[max_val_index(test_example_x[l])]
   outstr = outstr + "<"
   print outstr

   outstr = "Generated Text: >"
   for l in range(0,40):
       o,h = network.forward_pass(last_output)
       outstr = outstr + reverse_dictionary[max_val_index(o)]
       last_output = o
   outstr = outstr + "<"
   print outstr       


       
