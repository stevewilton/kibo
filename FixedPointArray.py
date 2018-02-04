   import math


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



# One of the following should be set:

MODE_STRICT_WITH_ERROR_CHECKING = 0
MODE_STRICT_WITHOUT_ERROR_CHECKING = 1
MODE_RELAXED_WITH_ERROR_CHECKING = 0

# Strict Mode.  In Strict mode, all operands must be of the same
# type, and any conversions between types need to be done explicitly,
# using resize function.  In Non-strict mode, you can do things like
#     a = b + 1.2 
# and the constant 1.2 will be converted to fixed point.  In non-strict
# mode, you can also do arithmetic on fixed point values of different sizes;
# in that case, the result will be the size of the first operand
#
# Error checking: checks that the input types are reasonable.  Good for
# debugging, but slows things down.  
#
# Note: the fastest is MODE_STRICT_WITHOUT_ERROR_CHECKING

assert(MODE_STRICT_WITH_ERROR_CHECKING + MODE_STRICT_WITHOUT_ERROR_CHECKING + MODE_RELAXED_WITH_ERROR_CHECKING == 1), "Must select exactly one Mode"



# Defaults.  

FRAC_BITS = 16
INT_BITS = 8


class FixedPointArray:
 
   def clip(self, x):
      """ Clip a value if it lies outside the allowable range"""
      return max(self.min_value,min(self.max_value, x))
      
   def quantize(self, value):
      """ quantize a floating point value """
      encoded_value = self.clip(round(value * (1<< self.frac_bits)))
      return ( float(encoded_value) / (1 << self.frac_bits) )

   def quantize_array(self):
      for i in range(0,self.values.shape[0]):
         for j in range(0,self.values.shape[1]):
            self.values[i][j] = self.quantize(self.values[i][j])

   def __init__(self, raw_array, int_bits=INT_BITS, frac_bits = FRAC_BITS):
      """ Create an Fixed Point object. """

      assert int_bits >= 1, "int_bits must be greater or equal than 1"
      assert frac_bits >= 0, "frac_bits must be greater or eqaul than 0"

      self.values = raw_array
      self.int_bits = int_bits
      self.frac_bits = frac_bits 
      self.max_value = (1<<(int_bits-1+frac_bits))-1
      self.min_value = -(1<<(int_bits-1+frac_bits))

#      self.quantize_array()


   def print_array(self):
      print(self.values) 

   def __add__(self, b):
      c = FixedPointArray(self.values + b.values, self.int_bits, self.frac_bits)
#      c.quantize_array()
      return(c)
