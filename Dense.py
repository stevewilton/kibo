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

    def save_weights_and_biases(self,directoryName,level):
       np.save(directoryName+"/level"+str(level)+"_W", self.W)
       np.save(directoryName+"/level"+str(level)+"_B", self.B)

    def load_weights_and_biases(self,directoryName,level):
       self.W = np.load(directoryName+"/level"+str(level)+"_W.npy")
       self.B = np.load(directoryName+"/level"+str(level)+"_B.npy")

    def __init__(self, num_inputs, num_outputs):

#       Initialize weights

       print ("In Dense constructor")

       self.num_inputs = num_inputs
       self.num_outputs = num_outputs

#       Initialize the following weight and bias matrices

       self.W = return_random_np_subarray(num_outputs, num_inputs)
       self.B = return_constant_np_subarray(num_outputs, 1, -1)

#    def preturb_weight(self, y_index, x_index, delta):
#       self.W[y_index][x_index] = self.W[y_index][x_index] + delta

    def forward_pass(self, x):
        self.inputs = x
        self.z = mydot(self.W, np.array(x)) + self.B
        self.a = tanh(self.z)
        #self.a = sigmoid(self.z)
        return (self.a)

    def get_deltas(self, deltas_from_previous_layer):
        self.deltas = deltas_from_previous_layer * tanh_prime(self.z)
        #self.deltas = deltas_from_previous_layer * sigmoid_prime(self.z)
        deltas_to_send_to_next_layer = mydot(self.W.T, self.deltas)
        return(deltas_to_send_to_next_layer)

    def update_weights(self, learning_rate):
       self.W = self.W - learning_rate * mydot(self.deltas,  np.array(self.inputs).T)
       self.B = self.B - learning_rate * self.deltas

#    def get_dcdw(self, yindex, xindex):
#       m = np.dot(self.deltas, self.inputs.T)
#       return m[yindex][xindex]
