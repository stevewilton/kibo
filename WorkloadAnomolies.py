'''
Workload which is a string of characters from a file
'''

import numpy as np
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from Settings import *

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


def WorkloadAnomolies():

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
