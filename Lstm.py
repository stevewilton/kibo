'''
An implementation of a single LSTM cell
'''

import numpy as np
import random
import collections
import time
from FixedPoint import FixedPoint
import math
from Settings import *
from Misc import *


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

    def __init__(self, num_inputs, num_outputs):

#       Initialize weights

       print ("In LSTM constructor")

       num_hidden = num_outputs  # same number of hidden units as outputs
       self.num_inputs = num_inputs
       self.num_outputs = num_outputs
       self.num_hidden = num_hidden

#       Initialize the following weight and bias matrices
#       Note that the bias for the forget is special, we initialize it all to 1

       self.Wf = return_random_np_subarray(num_hidden, num_inputs + num_outputs)
       self.Bf = return_one_np_subarray(num_hidden, 1)
       self.Wi = return_random_np_subarray(num_hidden, num_inputs + num_outputs)
       self.Bi = return_random_np_subarray(num_hidden, 1)
       self.Wc = return_random_np_subarray(num_hidden, num_inputs + num_outputs)
       self.Bc = return_random_np_subarray(num_hidden, 1)
       self.Wo = return_random_np_subarray(num_outputs, num_inputs + num_outputs)
       self.Bo = return_random_np_subarray(num_outputs, 1)

       # initialize internal state
       self.Ct = return_constant_np_subarray(num_hidden,1, 0.0)

       # initialize previous output
       self.ht = return_constant_np_subarray(num_outputs,1, 0.0)



    def reset(self):

#      Keep weights the same, but reset internal state

       # initialize internal state
       self.Ct = return_constant_np_subarray(self.num_hidden,1, 0.0)

       # initialize previous output
       self.ht = return_constant_np_subarray(self.num_outputs,1, 0.0)




#   This assumes you've recently done a forward pass.  We don't call it
#   here again, because if we do a second forward pass by accident ,the
#   state will change.  

    def get_deltas(self, x, dh):

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
