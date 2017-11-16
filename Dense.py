'''
An implementation of a dense (fully connected) layer
'''

import numpy as np
import random
import collections
import time
from FixedPoint import FixedPoint
import math
from Settings import *
from Misc import *

class Dense(object):

    def __repr__(self):
       return("Dense Layer with %d inputs, %d outputs" % (
                      self.num_inputs, self.num_outputs))


    def __str__(self):
       return("Dense Layer with %d inputs, %d outputs" % (
                      self.num_inputs, self.num_outputs))

    def print_weights_and_biases(self):
       print ("W:")
       print (self.W)
       print ("\nB:")
       print (self.B)

    def __init__(self, num_inputs, num_outputs):

#       Initialize weights

       print ("In Dense constructor")

       self.num_inputs = num_inputs
       self.num_outputs = num_outputs

#       Initialize the following weight and bias matrices

       self.W = return_random_np_subarray(num_outputs, num_inputs)
       self.B = return_constant_np_subarray(num_outputs, 1, -1)

    def preturb_weight(self, y_index, x_index, delta):
       self.W[y_index][x_index] = self.W[y_index][x_index] + delta

    def forward_pass(self, x):
        ft = sigmoid(mydot(self.W, np.array(x)) + self.B)
        self.inputs = x
        return (ft)

    def get_deltas(self, y):
       z = mydot(self.W, self.inputs) + self.B
       a = sigmoid(z)
#       error = 0.5*np.sum( (a-y)*(a-y) )
#       print error
       self.delta_outputs = (a - y) * sigmoid_prime(z)

       
       self.delta_inputs = mydot(self.W.T,self.delta_outputs)
       return(self.delta_inputs)

    def update_weights(self, learning_rate):
       self.W = self.W - learning_rate * mydot(self.delta_outputs, self.inputs.T)
       self.B = self.B - learning_rate * self.delta_outputs

    def get_dcdw(self, yindex, xindex):
       m = np.dot(self.delta_outputs, self.inputs.T)
       return m[yindex][xindex]
