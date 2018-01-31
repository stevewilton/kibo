'''
This is an example of a simple multi-layer perceptron network.
To make this as simple as possible, we use an XOR workload.  A single
Neuron won't train well on an XOR, however, a multi-layer network will.
In the code below, we do both.
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



def train_and_evaluate_xor_network(network, training_set, evaluation_set):
   print "   Training Network:"
   num_iters = 100000

   for iter in range(0,num_iters):
         total_error = 0
         training_example = random.choice(training_set)
#      for training_example in training_set:
         o = network.train(training_example[0], training_example[1], 0.01)      
         total_error = total_error + o
#      if (iter % 100 == 0):
#         print "      Iteration %d, average error = %f" % (iter, total_error / len(training_set_floating_point) )

   print "   Evaluating Network:"
   total_error = 0.0
   for evaluation_example in evaluation_set:
       o = network.forward_pass(evaluation_example[0])    
       print evaluation_example[1][0], o[0][0]
       if (( evaluation_example[1][0] > 0.5) == (o[0][0]>0.5) ):
           total_error = total_error + 1.0
   print "     Accuracy = %f" % (total_error / len(evaluation_set_floating_point))




print "Simple test for training a network to solve the XOR problem"

training_set_floating_point = [ [ [ [0.],[0.]], [0.]], 
                                [ [ [0.],[1.]], [1.]],
                                [ [ [1.],[0.]], [1.]],
                                [ [ [1.],[1.]], [0.]] ];
evaluation_set_floating_point = training_set_floating_point

print " "
print "Example 1.  Start with a single neuron.  We would not expect"
print "that this would train well, even for floating point"
print " "
print "Initializing Network"
network = Network(2)
network.add_level(0, 1)
train_and_evaluate_xor_network(network, training_set_floating_point, evaluation_set_floating_point)


print " "
print "Example 2.  Two layer network.  We would expect that this would work well"
print " "
print "Initializing Network"
network = Network(2)
network.add_level(0, 2)
network.add_level(0, 1)
train_and_evaluate_xor_network(network, training_set_floating_point, evaluation_set_floating_point)


