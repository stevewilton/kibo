'''
Workload which is a string of characters from a file
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
from Misc import *
from ArrayUtilities import *

num_training_examples = 500000
num_validation_examples = 10
num_test_examples = 50000
min_example_length = 90
max_example_length = 180

def convert_to_binary(c):
   binstr = ' '.join(format(ord(x), 'b') for x in c)
   if FIXED_POINT:
      binlist = [ [FixedPoint(float(x))] for x in binstr ]   
   else:
      binlist = [ [float(x)] for x in binstr ]   
   return binlist


def WorkloadAnomalies():

   consonants = "bcdfghjklmnpqrstvwxz"
   vowels = "aeiouy"

   training_set_x = []
   training_set_y = []
   validation_set_x = []
   validation_set_y = []
   test_set_x = []
   test_set_y = []

   for i in range(0,num_training_examples):
      cpos = random.choice(consonants)
      training_set_x.append(convert_to_binary(cpos) )
      # this is unsupervised, so no training set y
      training_set_y.append(0)

   for i in range(0,num_test_examples):
      if (random.randint(1,8) == 1):
         cpos = random.choice(vowels)
         y = 1
      else:
         cpos = random.choice(consonants)
         y = 0
      test_set_x.append(convert_to_binary(cpos) )
      test_set_y.append( y )

   for i in range(0,num_validation_examples):
      if (random.randint(1,8) == 1):
         cpos = random.choice(vowels)
         y = 1
      else:
         cpos = random.choice(consonants)
         y = 0
      validation_set_x.append(convert_to_binary(cpos) )
      validation_set_y.append( y )


   training_set = zip(training_set_x, training_set_y)
   validation_set = zip(validation_set_x, validation_set_y)
   test_set = zip(test_set_x, test_set_y)

   vocab_size = 7
   return((training_set, validation_set, test_set, vocab_size))
