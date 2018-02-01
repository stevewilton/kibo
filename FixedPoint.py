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
MODE_STRICT_WITHOUT_ERROR_CHECKING = 0
MODE_RELAXED_WITH_ERROR_CHECKING = 1

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


class FixedPoint:

   def __init__(self, value=0, int_bits=INT_BITS, frac_bits=FRAC_BITS):
      """ Create an Fixed Point object. """

      assert int_bits >= 1, "int_bits must be greater or equal than 1"
      assert frac_bits >= 0, "frac_bits must be greater or eqaul than 0"
      assert isinstance(value, int) | isinstance(value, float) | isinstance(value, FixedPoint), "initialized value type error"

      if (isinstance(value, FixedPoint)):
         value_to_store = value.val()
      else:
         value_to_store = value
            
      self.int_bits = int_bits
      self.frac_bits = frac_bits 
      self.max_value = (1<<(int_bits-1+frac_bits))-1
      self.min_value = -(1<<(int_bits-1+frac_bits))
      self.encoded = self.encode(value_to_store)

   def resize(self, int_bits, frac_bits):
      """ Convert the value to a new representation, but keep the value the same """

      assert int_bits >= 1, "int_bits must be greater or equal than 1"
      assert frac_bits >= 0, "frac_bits must be greater or equal than 0"

      stored_value = self.val()
      self.int_bits = int_bits
      self.frac_bits = frac_bits 
      self.max_value = (1<<(int_bits-1+frac_bits))-1
      self.min_value = -(1<<(int_bits-1+frac_bits))
      self.encoded = self.encode(stored_value)      
      
   def clip(self, x):
      """ Clip a value if it lies outside the allowable range"""
      return max(self.min_value,min(self.max_value, x))

   def encode(self, value):
      """ Convert a value to the encoded representation and clip it """
      return self.clip(round(value * (1<< self.frac_bits)))
                   
   def decode(self, encoded_value):
      """ Return the decoded value."""
      return float(encoded_value) / (1 << self.frac_bits)

   def val(self):
      """ The same as decode.  Provided to be more intuitive name """
      return float(self.encoded) / (1 << self.frac_bits)

   def __float__(self):
      """ Return the float (decoded) value """
      return float(self.encoded) / (1 << self.frac_bits)
   
   def __int__(self):
      """ Return the integer portion of the (decoded) value """    
      return int(self.val())
      
   def string_raw(self):
      """ Return the raw (encoded) value.  Normally only use this for debugging """
      return "x%x" % self.encoded

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

      if (MODE_RELAXED_WITH_ERROR_CHECKING):
         if (isinstance(b,int) | isinstance(b,float)):
            return(self.val() == b)

      # Otherwise, we are comparing two fixed point values
      return (self.val() == b.val())

   def __gt__(self,b):
      """ greather than comparison """

      if (MODE_RELAXED_WITH_ERROR_CHECKING):
         if (isinstance(b,int) | isinstance(b,float)):
            return(self.val() > b)
      return (self.val() > b.val())

   def __lt__(self,b):
      """ less than comparison """

      if (MODE_RELAXED_WITH_ERROR_CHECKING):
         if (isinstance(b,int) | isinstance(b,float)):
            return(self.val() < b)
      return (self.val() < b.val())

   def __ge__(self,b):
      """ greater or equal to """

      if (MODE_RELAXED_WITH_ERROR_CHECKING):
         if (isinstance(b,int) | isinstance(b,float)):
            return(self.val() >= b)
      return (self.val() >= b.val())

   def __le__(self,b):
      """ less than or equal to """

      if (MODE_RELAXED_WITH_ERROR_CHECKING):
         if (isinstance(b,int) | isinstance(b,float)):
            return(self.val() <= b)
      return (self.val() <= b.val())


   ########################################################################
   #
   #  Over-riding arithmetic functions. 
   #
   ########################################################################

   def __add__(self, b):
      """ add function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(self.encoded+b.encoded)
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(self.encoded+b.encoded)
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(self.val() + b, self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(self.val() + b.val(), self.int_bits, self.frac_bits)
      return retval


   def __radd__(self, b):
      """ radd function """
      return (self.__add__(b) )


   def __sub__(self, b):
      """ subtract function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(self.encoded-b.encoded)
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(self.encoded-b.encoded)
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(self.val() - b, self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(self.val() - b.val(), self.int_bits, self.frac_bits)
      return retval

   def __rsub__(self,b):
      """ rsub function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(b.encoded-self.encoded)
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(b.encoded-self.encoded)
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(b - self.val(), self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(b.val() - self.val(), self.int_bits, self.frac_bits)
      return retval

   def __iadd__(self, b):
      """ __iadd__ :  adding using the += operator """
      return(self.__add__(b))

   def __isub__(self, b):
      """ __isub__ :  adding using the -= operator """
      return(self.__sub__(b))

   def __neg__(self):
      """ take the negative of a fixed point value """
      retval = FixedPoint(0,self.int_bits, self.frac_bits)  - self
      return retval

#  probably never want to use this, since things may not be exactly 0,
#  but it is here just in case.

   def __bool__(self):
      return( self.encoded != 0) 


   def __mul__(self, b):
      """ multiply function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip((self.encoded*b.encoded)/ (1<<self.frac_bits))
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip((self.encoded*b.encoded)/ (1<<self.frac_bits))
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(self.val() * b, self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(self.val() * b.val(), self.int_bits, self.frac_bits)
      return retval

   def __rmul__(self, b):
      """ rmul function """
      return (self.__mul__(b) )

   def __imul__(self, b):
      """ __imul__ :  adding using the *= operator """
      return(self.__mul__(b))


   def __div__(self, b):
      """ divide function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(round((self.encoded * (1<< self.frac_bits)) / b.encoded))
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(round((self.encoded * (1<< self.frac_bits)) / b.encoded))
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(float(self.val()) / float(b), self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(float(self.val()) / float(b.val()), self.int_bits, self.frac_bits)
      return retval


   def __rdiv__(self, b):
      """ reverse divide function """

      if MODE_STRICT_WITHOUT_ERROR_CHECKING:
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(round((b.encoded * (1<< self.frac_bits)) / self.encoded))
         return retval

      if MODE_STRICT_WITH_ERROR_CHECKING:
         assert(b.int_bits == self.int_bits) & (b.frac_bits == self.frac_bits), "Strict mode: operands not aligned"
         retval = FixedPoint(0,self.int_bits, self.frac_bits) 
         retval.encoded = retval.clip(round((b.encoded * (1<< self.frac_bits)) / self.encoded))
         return retval

      if (isinstance(b,int) | isinstance(b,float)):
         retval = FixedPoint(float(b) / float(self.val()), self.int_bits, self.frac_bits)
      else:
         retval = FixedPoint(float(b.val()) / float(self.val()), self.int_bits, self.frac_bits)
      return retval

   def __idiv__(self, b):
      """ __idiv__ :  adding using the /= operator """
      return(self.__div__(b))


   def exp(a):
      """ exponent """
      if (isinstance(a,int) | isinstance(a,float)):
         return(math.exp(a))
      else:
         return( FixedPoint( math.exp(a), a.int_bits, a.frac_bits) )

   def tanh(a):
      """ tanh """
      if (isinstance(a,int) | isinstance(a,float)):
         return(math.tanh(a))
      else:
         return( FixedPoint( math.tanh(a), a.int_bits, a.frac_bits) )


