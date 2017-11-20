import math
from Settings import *

class FixedPoint:

   def clip(self, x):
      return max(self.min_value,min(self.max_value, x))


   def encode(self, value):
         return self.clip(round(value * (1<< self.frac_bits)))
                   
   def decode(self, encoded_value):
         return encoded_value / (1 << self.frac_bits)

   def val(self):
         return self.encoded / (1 << self.frac_bits)

   def __float__(self):
         return self.encoded / (1 << self.frac_bits)
   
   def __init__(self, value=0, int_bits=INT_BITS, frac_bits=FRAC_BITS):
      self.int_bits = int_bits
      self.frac_bits = frac_bits 
      self.max_value = (1<<(int_bits-1+frac_bits))-1
      self.min_value = -(1<<(int_bits-1+frac_bits))
      if (value == 0):
         self.encoded = 0
      else:
         self.encoded = self.encode(value)
      
   def string_raw(self):
      return "%x" % self.encoded

   def print_val(self):
      return "%f" % self.decode(self.encoded)

   def __str__(self):
      return "%f" % self.decode(self.encoded)

   def __repr__(self):
      return "%f" % self.decode(self.encoded)
                     
   def print_info(self):
      print "encoded = 0x%x" % self.encoded
      print "int_bits = %d" % self.int_bits
      print "frac_bits = %d" % self.frac_bits
      print "max_value = 0x%x" % self.max_value
      print "min_value = 0x%x" % self.min_value
      print "decoded value = %f" % self.decode(self.encoded)

   def __iadd__(self, b):
      print "IN IADD"
   def __imul__(self, b):
      print "IN IMUL"


   def __eq__(self,b):
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.encoded == b)

      return (self.encoded == b.encoded)

   def __gt__(self,b):
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.encoded > b)

      return (self.encoded > b.encoded)

   def __lt__(self,b):
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.encoded < b)
      return (self.encoded < b.encoded)

   def __ge__(self,b):
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.encoded >= b)

      return (self.encoded >= b.encoded)

   def __le__(self,b):
      if (isinstance(b,int) | isinstance(b,float)):
         return(self.encoded <= b)

      return (self.encoded <= b.encoded)

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


