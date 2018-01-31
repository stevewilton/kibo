import math
from Settings import *

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


# Defaults

FRAC_BITS = 16
INT_BITS = 8

class FixedPoint:

   def __init__(self, value=0, int_bits=INT_BITS, frac_bits=FRAC_BITS):
      """ Create an Fixed Point object. """

      assert int_bits > 0, "int_bits must be greater than 0"
      assert frac_bits > 0, "frac_bits must be greater than 0"
      
      self.int_bits = int_bits
      self.frac_bits = frac_bits 
      self.max_value = (1<<(int_bits-1+frac_bits))-1
      self.min_value = -(1<<(int_bits-1+frac_bits))
      if (value == 0):
         self.encoded = 0
      else:
         self.encoded = self.encode(value)

   def clip(self, x):
      """ Clip a value if it lies outside the allowable range"""
      return max(self.min_value,min(self.max_value, x))

   def encode(self, value):
      """ Convert a value to the encoded representation and clip it """
      return self.clip(round(value * (1<< self.frac_bits)))
                   
   def decode(self, encoded_value):
      """ Return the decoded value."""
      return encoded_value / (1 << self.frac_bits)

   def val(self):
      """ The same as decode.  Provided to be more intuitive name """
      return self.encoded / (1 << self.frac_bits)

   def __float__(self):
      """ Return the float (decoded) value """
      return self.encoded / (1 << self.frac_bits)
   
   def __int__(self):
      """ Return the integer portion of the (decoded) value """    
      return int(self.val())
      
   def string_raw(self):
      """ Return the raw (encoded) value.  Normally only use this for debugging """
      return "0x%x" % self.encoded

   def __str__(self):
      """ string representation.  Include the size and the decoded value """      """ string representation.  Include the size and the decoded value """
      return "<%d,%d>%f" % (self.int_bits,self.frac_bits,self.decode(self.encoded))

   def __repr__(self):
      """ string representation.  Include the size and the decoded value """
      return "<%d,%d>%f" % (self.int_bits,self.frac_bits,self.decode(self.encoded))
                     
   def info(self):
      """ Print information about the object in long form.  Useful for debugging """
      print "encoded = 0x%x" % self.encoded
      print "int_bits = %d" % self.int_bits
      print "frac_bits = %d" % self.frac_bits
      print "max_value = 0x%x" % self.max_value
      print "min_value = 0x%x" % self.min_value
      print "decoded value = %f" % self.decode(self.encoded)


   #######################################################################
   # Over-ride comparison functions.  Note that these comparison functions
   # allow for a comparison of a fixed point to a constant value, or two
   # fixed point numbers.  In all cases, it is the decoded value that is
   # compared.  So, if two numbers have different representations, but encode
   # the same value, they will be equal.  Warning: different representations
   # may lead to different rounding behaviour
   #########################################################################

   def __eq__(self,b):
      """ equality comparison"""

      # See if we are comparing to a fixed integer or floating point number

      if (isinstance(b,int) | isinstance(b,float)):
         return(self.val() == b)

      # Otherwise, we are comparing two fixed point values
      return (self.val() == b.val())

   def __gt__(self,b):
      """ greather than comparison """
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.val() > b)
      return (self.val() > b.val())

   def __lt__(self,b):
      """ less than comparison """
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.val() < b)
      return (self.val() < b.val())

   def __ge__(self,b):
      """ greater or equal to """
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.val() >= b)
      return (self.val() >= b.val())

   def __le__(self,b):
      """ less than or equal to """
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.val() <= b)
      return (self.val() <= b.val())


   # 

   def __add__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = self + FixedPoint(b)
         return retval

#      if ((b.int_bits != self.int_bits) | (b.frac_bits != self.frac_bits)):
#         print "Error in addition: adding two numbers with different reps"
#      else:
         
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(self.encoded+b.encoded)
      return retval

   def __radd__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(b) + self
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in addition: adding two numbers with different reps"
#      else:
         
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(self.encoded+b.encoded)
      return retval

   def __sub__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = self - FixedPoint(b)
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in addition: subtracting two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(self.encoded-b.encoded)
      return retval


   def __iadd__(self, b):
      print "IN IADD"

   def __imul__(self, b):
      print "IN IMUL"


   def __rsub__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(b) - self
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in addition: subtracting two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(self.encoded-b.encoded)
      return retval


   def __neg__(self):
         retval = FixedPoint(0,self.int_bits, self.frac_bits)  - self
         return retval

   def __mul__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = self * FixedPoint(b)
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in multiply: multiplying two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip((self.encoded*b.encoded)/ (1<<self.frac_bits))
      return retval

   def __rmul__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = self * FixedPoint(b)
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#        print "Error in multiply: multiplying two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip((self.encoded*b.encoded)/ (1<<self.frac_bits))
      return retval

#  probably never want to use this, since things may not be exactly 0,
#  but it is here just in case.

   def __bool__(self):
      return( self.encoded != 0) 

   def __div__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = self / FixedPoint(b)
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in divide: multiplying two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(round((self.encoded * (1<< self.frac_bits)) / b.encoded))
      return retval

   def __rdiv__(self, b):
      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(b) /self
         return retval

#      if ((b.int_bits != self.int_bits) |
#          (b.frac_bits != self.frac_bits)):
#         print "Error in divide: multiplying two numbers with different reps"
#      else:
      retval = FixedPoint(0,self.int_bits, self.frac_bits) 
      retval.encoded = retval.clip(round((self.encoded * (1<< self.frac_bits)) / b.encoded))
      return retval


   def exp(a):
      if (isinstance(a,int) | isinstance(a,float)):
         return(math.exp(a))
      else:
         return( FixedPoint( math.exp(a), a.int_bits, a.frac_bits) )

   def tanh(a):
      if (isinstance(a,int) | isinstance(a,float)):
         return(math.tanh(a))
      else:
         return( FixedPoint( math.tanh(a), a.int_bits, a.frac_bits) )


