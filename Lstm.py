'''
An implementation of a single LSTM cell
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


class LSTM(object):

    def __repr__(self):
       return("LSTM with %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))


    def __str__(self):
       return("LSTM with %d inputs, %d hidden units, %d outputs" % (
                      self.num_inputs, self.num_hidden, self.num_outputs))

    def print_weights_and_biases(self):
       print ("Wf:")
       print (self.Wf)
       print ("\nBf:")
       print (self.Bf)
       print ("Wi:")
       print (self.Wi)
       print ("\nBi:")
       print (self.Bi)
       print ("Wc:")
       print (self.Wc)
       print ("\nBc:")
       print (self.Bc)
       print ("Wo:")
       print (self.Wo)
       print ("\nBo:")
       print (self.Bo)

    def save_weights_and_biases(self,directoryName,level):
       np.save(directoryName+"/level"+str(level)+"_Wf", self.Wf)
       np.save(directoryName+"/level"+str(level)+"_Bf", self.Bf)
       np.save(directoryName+"/level"+str(level)+"_Wi", self.Wi)
       np.save(directoryName+"/level"+str(level)+"_Bi", self.Bi)
       np.save(directoryName+"/level"+str(level)+"_Wc", self.Wc)
       np.save(directoryName+"/level"+str(level)+"_Bc", self.Bc)
       np.save(directoryName+"/level"+str(level)+"_Wo", self.Wo)
       np.save(directoryName+"/level"+str(level)+"_Bo", self.Bo)

    def load_weights_and_biases(self,directoryName,level):
       self.Wf = np.load(directoryName+"/level"+str(level)+"_Wf.npy")
       self.Bf = np.load(directoryName+"/level"+str(level)+"_Bf.npy")
       self.Wi = np.load(directoryName+"/level"+str(level)+"_Wi.npy")
       self.Bi = np.load(directoryName+"/level"+str(level)+"_Bi.npy")
       self.Wc = np.load(directoryName+"/level"+str(level)+"_Wc.npy")
       self.Bc = np.load(directoryName+"/level"+str(level)+"_Bc.npy")
       self.Wo = np.load(directoryName+"/level"+str(level)+"_Wo.npy")
       self.Bo = np.load(directoryName+"/level"+str(level)+"_Bo.npy")
       if self.fixed_point:
           self.Wf = convert_array_to_fixed(self.Wf,self.Wf.shape[0],self.Wf.shape[1], self.int_bits, self.frac_bits)
           self.Bf = convert_array_to_fixed(self.Bf,self.Bf.shape[0],self.Bf.shape[1], self.int_bits, self.frac_bits)
           self.Wi = convert_array_to_fixed(self.Wi,self.Wi.shape[0],self.Wi.shape[1], self.int_bits, self.frac_bits)
           self.Bi = convert_array_to_fixed(self.Bi,self.Bi.shape[0],self.Bi.shape[1], self.int_bits, self.frac_bits)
           self.Wc = convert_array_to_fixed(self.Wc,self.Wc.shape[0],self.Wc.shape[1], self.int_bits, self.frac_bits)
           self.Bc = convert_array_to_fixed(self.Bc,self.Bc.shape[0],self.Bc.shape[1], self.int_bits, self.frac_bits)
           self.Wo = convert_array_to_fixed(self.Wo,self.Wo.shape[0],self.Wo.shape[1], self.int_bits, self.frac_bits)
           self.Bo = convert_array_to_fixed(self.Bo,self.Bo.shape[0],self.Bo.shape[1], self.int_bits, self.frac_bits)

    def print_internal_state(self):
       print ("\nC:")
       print (self.C)

    def preturb_weight(self, matrix, y_index, x_index, delta):
       if (matrix == 2):  
	  self.Wf[y_index][x_index] = self.Wf[y_index][x_index] + delta
       elif (matrix == 3):
	  self.Wi[y_index][x_index] = self.Wi[y_index][x_index] + delta
       elif (matrix == 4):
	  self.Wc[y_index][x_index] = self.Wc[y_index][x_index] + delta
       elif (matrix == 5):
	  self.Wo[y_index][x_index] = self.Wo[y_index][x_index] + delta
       else:
          print "Dont know how to preturb that matrix"

    def __init__(self, num_inputs, num_outputs, fixed_point = 0, int_bits = 0, frac_bits = 0):

#       Initialize weights

       print ("In LSTM constructor")

       num_hidden = num_outputs  # same number of hidden units as outputs
       self.num_inputs = num_inputs
       self.num_outputs = num_outputs
       self.num_hidden = num_hidden
       self.fixed_point = fixed_point
       self.int_bits = int_bits
       self.frac_bits = frac_bits

#       Initialize the following weight and bias matrices
#       Note that the bias for the forget is special, we initialize it all to 1

       self.Wf = return_random_np_subarray(num_hidden, num_inputs + num_outputs, fixed_point, int_bits, frac_bits)
       self.Bf = return_one_np_subarray(num_hidden, 1, fixed_point, int_bits, frac_bits)
       self.Wi = return_random_np_subarray(num_hidden, num_inputs + num_outputs, fixed_point, int_bits, frac_bits)
       self.Bi = return_random_np_subarray(num_hidden, 1, fixed_point, int_bits, frac_bits)
       self.Wc = return_random_np_subarray(num_hidden, num_inputs + num_outputs, fixed_point, int_bits, frac_bits)
       self.Bc = return_random_np_subarray(num_hidden, 1, fixed_point, int_bits, frac_bits)
       self.Wo = return_random_np_subarray(num_outputs, num_inputs + num_outputs, fixed_point, int_bits, frac_bits)
       self.Bo = return_random_np_subarray(num_outputs, 1, fixed_point, int_bits, frac_bits)

       # initialize internal state
       self.Ct = return_constant_np_subarray(num_hidden,1, 0.0, fixed_point, int_bits, frac_bits)

       # initialize previous output
       self.ht = return_constant_np_subarray(num_outputs,1, 0.0, fixed_point, int_bits, frac_bits)



    def reset(self):

#      Keep weights the same, but reset internal state

       # initialize internal state
       self.Ct = return_constant_np_subarray(self.num_hidden,1, 0.0, self.fixed_point, self.int_bits, self.frac_bits)

       # initialize previous output
       self.ht = return_constant_np_subarray(self.num_outputs,1, 0.0, self.fixed_point, self.int_bits, self.frac_bits)




#   This assumes you've recently done a forward pass.  We don't call it
#   here again, because if we do a second forward pass by accident ,the
#   state will change.  

    def get_deltas(self, dh):

        self.do = dh * np.tanh(self.Ct)

        # Note: since we are only going back one step, don't need to
        # accumulate anything here

        self.dc = dh * self.ot * tanh_prime(self.Ct)
        self.di = self.dc * self.Ctt
        self.df = self.dc * self.Ctm1

        self.dCtt = self.dc * self.it
        self.dCtm1 = self.dc * self.ft
        self.dCtthat = self.dCtt * ( 1 - np.tanh(self.Ctt)*np.tanh(self.Ctt) )
        self.dihat = self.di * self.it * (1 - self.it)
        self.dfhat = self.df * self.ft * (1 - self.ft)
        self.dohat = self.do * self.ot * (1 - self.ot)

        # we should send the deltas for x but this hasn't been implemented yet
        return(0.0)


    def update_weights(self, learning_rate):
       u = np.concatenate((self.htm1,self.inputs))
       self.Wf = self.Wf - learning_rate * mydot(self.dfhat, u.T)
       self.Wi = self.Wi - learning_rate * mydot(self.dihat, u.T)
       self.Wc = self.Wc - learning_rate * mydot(self.dCtthat, u.T)
       self.Wo = self.Wo - learning_rate * mydot(self.dohat, u.T)
       self.Bf = self.Bf - learning_rate * self.dfhat
       self.Bi = self.Bi - learning_rate * self.dihat
       self.Bc = self.Bc - learning_rate * self.dCtthat
       self.Bo = self.Bo - learning_rate * self.dohat

       
    def forward_pass(self, x):        
        self.inputs = x
        self.htm1 = self.ht
        self.Ctm1 = self.Ct
        self.ft = sigmoid(mydot(self.Wf, np.concatenate((self.htm1,x))) + self.Bf)
        self.it = sigmoid(mydot(self.Wi, np.concatenate((self.htm1,x))) + self.Bi)
        self.Ctt = np.tanh(mydot(self.Wc, np.concatenate((self.htm1,x))) + self.Bc)
        self.Ct = self.ft * self.Ctm1 + self.it * self.Ctt
        self.ot = sigmoid(mydot(self.Wo, np.concatenate((self.htm1,x))) + self.Bo)
        self.ht = self.ot * np.tanh(self.Ct)

        # save information we will need during the backwards pass
        # note: we only save one copy, because we are assuming on-line
        # learning where we only go back one step.  If we were to go back
        # multiple steps, we'd have to remember several steps of this 
        # information.

        return(self.ht)


    def get_dcdw(self, matrix, yindex, xindex):
       u = np.concatenate((self.htm1,self.inputs))
       if (matrix == 2):
          m = np.dot(self.dfhat, u.T)
       elif (matrix == 3):
          m = np.dot(self.dihat, u.T)
       elif (matrix == 4):
          m = np.dot(self.dCtthat, u.T)
       elif (matrix == 5):
          m = np.dot(self.dohat, u.T)
       else:
          print "Dont know how to get deltas for that matrix"

       return m[yindex][xindex]
