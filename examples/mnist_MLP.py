'''
An example on how to use KIBO to classify time series data
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

# Remember to 
# 1 - Download MNIST here http://yann.lecun.com/exdb/mnist/
# 2 - Put it into a folder named mnist_data (unziped)
# 3 - Install parser using "pip install python-mnist"

import sys
sys.path.append('..')
import traceback
import pandas as pd
import scipy.io as sio
import os
import logging as logger
from FixedPoint import FixedPoint
from Lstm import LSTM
from Dense import Dense
from Network import *
import numpy as np
from mnist import MNIST 

FIXED_POINT=0

# Split an array into a array of arrays
def split(arr):
  return np.array([arr]).transpose().tolist()

# Map to a one hot encoding
def one_hot(val, len):
  if FIXED_POINT:
     retval = [[FixedPoint(0.) for col in range(1)] for row in range(len)]
     retval[val][0] = FixedPoint(1)
  else:
     retval = [[0. for col in range(1)] for row in range(len)]
     retval[val][0] = 1.0
  return(retval)

def train():
    total_error = 0

    # Shuffling training examples
    shuffleEqually(imagesTraining,labelsTraining)

    # Loop though all training data
    for i in range(len(imagesTraining)):

       #Converting the output to one_hot
       imageLabel = one_hot(labelsTraining[i],10)

       # Train network
       o = network.train(imagesTraining[i], imageLabel, 0.001)

       # Save the error only for the last batch
       total_error = total_error + o
       

    print("\tAverage loss = "+str(total_error/len(imagesTraining)))


def getAccuracy():
    # Variables used for counting good predictions
    hit=0.

    # Loop through all elements of test set
    for i in range(len(imagesTest)):
       #Converting the output to one_hot
       imageLabel = one_hot(labelsTest[i],10)

       # Get loss from forward pass
       o = network.forward_pass(imagesTest[i])
       
       # Check if our prediction is accurate
       correctLabel=np.argmax(imageLabel)
       netWorkLabel=np.argmax(o)
       if correctLabel==netWorkLabel:
          hit=hit+1

    print ("\tAccuracy = " +str(hit/len(imagesTest)*100 ) +"%")

def shuffleEqually(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

# Some parameters for the training and
inputSize=28*28
maxEpochs=1000
num_inputs=inputSize

# Dataset construction
print "Loading data set..."
mndata = MNIST('mnist_data/')
imagesTraining, labelsTraining = mndata.load_training()
imagesTest, labelsTest = mndata.load_testing()

# Processing dataset
print "Processing data set..."
imagesTraining = np.array([split(x) for x in imagesTraining])/255.
imagesTest = np.array([split(x) for x in imagesTest])/255.

# Network initialization
print "Initializing Network"
denseLayer=0
lstmLayer=1
network = Network(num_inputs)
network.add_level(denseLayer, 100,'tanh')
network.add_level(denseLayer, 10,'softmax') 

for epoch in range(maxEpochs):
    print("Epoch "+str(epoch)+"/"+str(maxEpochs))
    getAccuracy()
    train()
    if epoch%10==0:
        network.save_weights_and_biases("mnist_epoch_"+str(epoch))
    
