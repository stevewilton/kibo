'''
Run file
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
import math
from FixedPoint import FixedPoint
from Lstm import LSTM
from Dense import Dense
from Network import *
from WorkloadAnomalies import *
from Misc import *
from ArrayUtilities import *

FIXED_POINT = 0
INT_BITS = 8
FRAC_BITS = 8

print "Constructing data set"

training_set, validaton_set, test_set, vocab_size = WorkloadAnomalies(FIXED_POINT)


print "Initializing Network"



network = Network(vocab_size, FIXED_POINT, INT_BITS, FRAC_BITS)
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

 








       
