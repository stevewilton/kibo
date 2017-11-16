'''
An implementation of a dense (fully connected) layer
'''

import numpy as np
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from Settings import *
from Misc import *
from Dense import *
from Lstm import *

class Network(object):

    def __repr__(self):
       if self.include_dense:
          return("Network consisting of one LSTM cell followed by one fully connected later.  Network has %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))
       else:
          return("Network consisting of one LSTM cell.  Network has %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))


    def __str__(self):
       if self.include_dense:
          return("Network consisting of one LSTM cell followed by one fully connected later.  Network has %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))
       else:
          return("Network consisting of one LSTM cell.  Network has %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))

    def print_weights_and_biases(self):
       print ("LSTM Cell:")
       self.lstm.print_weights_and_biases()
       if self.include_dense:
          print ("Fully Connected Layer:")
          self.dense.print_weights_and_biases() 

    def reset(self):
       self.lstm.reset()
       # nothing to reset in the Dense layer

    def __init__(self, num_inputs, num_hidden, num_outputs, include_dense=1):
       print ("In Network Constructor")

       if (include_dense == 0):
          if (num_outputs != num_hidden):
              print ("If there is no dense layer, the number of outputs must be equal to the number of hidden units")
              exit(0)

       self.num_inputs = num_inputs
       self.num_outputs = num_outputs
       self.num_hidden = num_hidden
       self.include_dense = include_dense

       # Right now, we only generate an LSTM followed by a FC layer
       # This can be generalized here

       self.lstm = LSTM(num_inputs,num_hidden)
       if (include_dense):
          self.dense = Dense(num_hidden,num_outputs)
       

    def preturb_weight(self, matrix, yindex, xindex, val):
       if self.include_dense == 0:
           print("Write now, preturb weight in network does not work on the lstm layer.")
           print("Call the appropriate routine in lstm object instead.  Or fix this.")
           exit(0)

       if (matrix == 1):
           self.dense.preturb_weight(yindex, xindex, val)
       else:
           print ("Dont know how to preturn that weight")


    def forward_pass(self, x):
       h = self.lstm.forward_pass(x)
       if (self.include_dense):
          o = self.dense.forward_pass(h)
       else:
          o = h
       return(o,h)

    def train(self, x, y, learning_rate):
       o,h = self.forward_pass(x)
       if (self.include_dense):
          dh = self.dense.get_deltas(y)
       else:
          dh = o-y

       self.lstm.get_deltas(x,dh)
  
       if self.include_dense:
          self.dense.update_weights(learning_rate)

       self.lstm.update_weights(learning_rate)      

