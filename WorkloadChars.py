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
from Settings import *

num_training_examples = 1000
num_validation_examples = 10
num_test_examples = 32
min_example_length = 90
max_example_length = 180


def max_val_index(f):
   np.reshape(f,(np.size(f)))
   return(np.argmax(f))

def flatten(l):
  out = ""
  for item in l:
    out = out + " " + item.strip()
  return [out]

def one_hot(val, len):
  if FIXED_POINT:
     retval = [[FixedPoint(-0.04) for col in range(1)] for row in range(len)]
     retval[val][0] = FixedPoint(1)
  else:
     retval = [[-0.04 for col in range(1)] for row in range(len)]
     retval[val][0] = 1.0
  return(retval)


def WorkloadChars():

   # read the original text

   training_file = 'words.txt'
   training_file = '../yy'

   with open(training_file) as f:
       content = flatten(f.readlines())
   content = content[0]

   # change it so it is all lower case letters and space
   # upper case is converted to lower case, all other characters
   # are omitted

   filtered_content = []
   for l in content:
#      if (l == " ") or l.islower():
      if l.islower():
         filtered_content.append(l)
      elif l.isupper():
         filtered_content.append(l.lower())

   # construct dictionaries for fast conversion

   set = {l for l in filtered_content}
   dictionary = dict()
   for f in set:
      dictionary[f] = len(dictionary)

   reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

   # cache for onehot values for each character

   vocab_size = len(dictionary)
   onehot = dict()
   for f in set:
      onehot[f] = one_hot(dictionary[f], vocab_size)

   # determine training, validation, and test vectors

   training_set_x = []
   training_set_y = []
   validation_set_x = []
   validation_set_y = []
   test_set_x = []
   test_set_y = []

   for i in range(0,num_training_examples):
      length = random.randint(min_example_length,max_example_length)
      start = random.randint(0, len(filtered_content)-1-length)
      training_set_x.append([onehot[c] for c in filtered_content[start:start+length-1]])
      training_set_y.append([onehot[c] for c in filtered_content[start+1:start+length]])      

   for i in range(0,num_validation_examples):
      length = random.randint(min_example_length,max_example_length)
      start = random.randint(0, len(filtered_content)-1-length)
      validation_set_x.append([onehot[c] for c in filtered_content[start:start+length-1]])
      validation_set_y.append([onehot[c] for c in filtered_content[start+1:start+length]])

   for i in range(0,num_test_examples):
      length = random.randint(min_example_length,max_example_length)
      start = random.randint(0, len(filtered_content)-1-length)
      test_set_x.append([onehot[c] for c in filtered_content[start:start+length-1]])
      test_set_y.append([onehot[c] for c in filtered_content[start+1:start+length]])

   training_set = zip(training_set_x, training_set_y)
   validation_set = zip(validation_set_x, validation_set_y)
   test_set = zip(test_set_x, test_set_y)

   return((training_set, validation_set, test_set, vocab_size, reverse_dictionary))
