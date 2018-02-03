'''
An implementation of a dense (fully connected) layer
'''

#-----------------------------------------------------------------------------
#
# Permission to use, copy, and modify this software and its documentation is
# hereby granted only under the following terms and conditions.  Both the
# above copyright notice and this permission notice must appear in all copies
# of the software, derivative works or modified versions, and any portions
# thereof, and both notices must appear in supporting documentation.
#
# This software may be distributed (but not offered for sale or transferred
# for compensation) to third parties, provided such third parties agree to
# abide by the terms and conditions of this notice.
# 
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHORS, AS
# WELL AS THE UNIVERSITY OF BRITISH COLUMBIA AND 
# THE UNIVERSITY OF SYDNEY DISCLAIM ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE, INCLUDING ALL IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS.   IN NO EVENT SHALL THE
# AUTHORS OR THE UNIVERSITY OF BRITISH COLUMBIA OR THE 
# UNIVERSITY OF SYDNEY BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR
# PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#---------------------------------------------------------------------------


import numpy as np
import random
import collections
import time
from FixedPoint import FixedPoint
import math
from Misc import *
from ArrayUtilities import *

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

    def __init__(self, num_inputs, num_outputs, activation, fixed_point = 0, int_bits = 0, frac_bits = 0):

#       Initialize weights

       print ("In Dense constructor")

       self.num_inputs = num_inputs
       self.num_outputs = num_outputs
       self.activation = activation

#       Initialize the following weight and bias matrices

       self.W = return_random_np_subarray(num_outputs, num_inputs, fixed_point, int_bits, frac_bits)
       self.B = return_constant_np_subarray(num_outputs, 1, -1, fixed_point, int_bits, frac_bits)

#    def preturb_weight(self, y_index, x_index, delta):
#       self.W[y_index][x_index] = self.W[y_index][x_index] + delta

    def forward_pass(self, x):
        self.inputs = x
        self.z = mydot(self.W, np.array(x)) + self.B
        if self.activation=='sigmoid':
          self.a = sigmoid(self.z)
        elif self.activation=='softmax':
          self.a = softmax(self.z)
        elif self.activation=='tanh':
          self.a = tanh(self.z)
        elif self.activation=='relu':
          self.a = relu(self.z)
        return (self.a)

    def get_deltas(self, deltas_from_previous_layer):
        if self.activation=='sigmoid':
          self.deltas = deltas_from_previous_layer * sigmoid_prime(self.z)
        elif self.activation=='softmax':
          self.deltas = deltas_from_previous_layer
        elif self.activation=='tanh':
          self.deltas = deltas_from_previous_layer * tanh_prime(self.z)
        elif self.activation=='relu':
          self.deltas = deltas_from_previous_layer * relu_prime(self.z)
          
        deltas_to_send_to_next_layer = mydot(self.W.T, self.deltas)
        return(deltas_to_send_to_next_layer)

    def update_weights(self, learning_rate):
       self.W = self.W - learning_rate * mydot(self.deltas,  np.array(self.inputs).T)
       self.B = self.B - learning_rate * self.deltas

#    def get_dcdw(self, yindex, xindex):
#       m = np.dot(self.deltas, self.inputs.T)
#       return m[yindex][xindex]
