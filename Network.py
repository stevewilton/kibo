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
       return("Network consisting of %d layers" % self.num_layers)

    def __str__(self):
       return("Network consisting of %d layers" % self.num_layers)

    def print_weights_and_biases(self):
       for i in range(0,self.num_levels):
           print "Level %d:" % i
           self.level[i].print_weights_and_biases()


#    def reset(self):
#       self.lstm.reset()
#       # nothing to reset in the Dense layer

    def __init__(self, num_inputs):
       print ("In Network Constructor")

       self.num_inputs = num_inputs
       self.num_levels = 0
       self.level_type = {}  
       self.level = {}
       self.outputs_from_last_level = num_inputs

    # type = 0 for dense and 1 for lstm
    # for a dense, the level size indicates the number of neurons in that level
    # for a lstm, the level size indicates the number of hidden units in that level
   
    def add_level(self, level_type, level_size):

       if (level_type == 0):
          # Dense level
          self.level_type [self.num_levels] = 0
          self.level [self.num_levels] = Dense(self.outputs_from_last_level, level_size)
          self.outputs_from_last_level = level_size
          self.num_outputs = level_size
          self.num_levels = self.num_levels + 1
       elif (level_type == 1):
          # LSTM Level
          self.level_type [self.num_levels] = 1
          self.level [self.num_levels] = LSTM(self.outputs_from_last_level, level_size)
          self.num_outputs = level_size
          self.outputs_from_last_level = level_size
          self.num_levels = self.num_levels + 1
       else:
          print ("Do not know how to add a level of type %d. Only Dense (0) and LSTM (1) are currently supported" % level_type)

       # Right now, we can only implement networks that contain an LSTM
       # if the LSTM is the first level.  This restriction coudl be fixed
       # by modifying lstm.get_deltas().  For now, jsut error out if this
       # sort of network is created.  We check here rather than in train()
       # just to speed up the trianing loop.

       for i in range(1,self.num_levels):
          if (self.level_type[i] == 1):
             print "You have created a network in which the LSTM is fed"
             print "by another layer.  Right now, training of this sort"
             print "of network is not allowed, since the code to return"
             print "the deltas coming out of the LSTM has not been "
             print "implemented.  This could be fixed by fixing lstm.get_deltas()"
             print "For now, you can use this network, but nothing below the LSTM"
             print "will be trained."
             

              

#    def preturb_weight(self, matrix, yindex, xindex, val):
#       if self.include_dense == 0:
#           print("Write now, preturb weight in network does not work on the lstm layer.")
#           print("Call the appropriate routine in lstm object instead.  Or fix this.")
#           exit(0)
#
#       if (matrix == 1):
#           self.dense.preturb_weight(yindex, xindex, val)
#       else:
#          print ("Dont know how to preturn that weight")


    def forward_pass(self, x):
       f = x
       for i in range(0,self.num_levels):
          f = self.level[i].forward_pass(f)
       return f

    def train(self, x, y, learning_rate):

       # do a forward pass to get outputs
       o = self.forward_pass(x)
       deltas_to_send_down = o-y

       for i in range(self.num_levels-1, -1, -1):
          deltas_to_send_down = self.level[i].get_deltas(deltas_to_send_down)

       # now update the weights
       for i in range(0,self.num_levels):
          self.level[i].update_weights(learning_rate)
          


