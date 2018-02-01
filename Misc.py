'''
Some Misc. functions that are useful 
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
from FixedPoint import *



def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1.0-sigmoid(z)) 

def tanh(z):
    return (1.0-np.exp(-2*z))/(1.0+np.exp(-2*z))

def tanh_prime(z): 
    return 1.0 - np.tanh(z) * np.tanh(z)

def get_cost(a, y):
    return 0.5*np.sum((a-y)*(a-y))

