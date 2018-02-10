'''
An example on how to use KIBO to learn time series data
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

print "This is a simple example. Try to make the window smaller and the training sample larger to make it more interesting"

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
import ecg_generation as ecg
import time

def train():
    total_error = 0
    trainCounter=0

    # Shuffling training examples
    np.random.shuffle(trainingSet)

    # Loop though all training data
    for i in range(len(trainingSet)):
       # Reset internal state of LSTM
       network.level[0].reset()

       #Build Internal State
       for window in internalStateWindow:
           network.forward_pass(trainingSet[i][window])

       #Train after having internal state
       for window in trainTestWindow:
           o = network.train(trainingSet[i][window],trainingSet[i][max(window)+1],0.1) 
           trainCounter=trainCounter+1
           total_error = total_error + o    

    print("\tAverage loss = "+str(total_error/trainCounter))


# Split an array into a array of arrays
def split(arr):
  return np.array([arr]).transpose().tolist()

def generateEcg(numSamples,timeLength):
  dataset = []
  for i in range(numSamples):
    dataset.append(split(ecg.generateEcg(bpm,timeLength,noiseSD,samplingRate)))
  return np.array(dataset)

def getAccuracy():

    # Loop through all elements of test set
    for i in range(len(testSet)):
       # Reset internal state of LSTM
       network.level[0].reset()

       #Create a shorter version of the test to append predictions to
       prediction=testSet[i][0:internalStateSamples]

       #Build up internal state
       for window in internalStateWindow:
           network.forward_pass(prediction[window])                                                                                                                                                                                                                                                                                                                                                        

       # Make predictions base on last elements
       for i in xrange(predictSamples):
           o = network.forward_pass(prediction[-windowSize:])
           prediction = np.concatenate([prediction,o])
       
       if showPlot:
           ecg.ecgInteractivePlot(prediction[0:internalStateSamples],prediction[internalStateSamples:-1],samplingRate,bpm)

# Some parameters for the training and generatig data
windowSize=40
maxEpochs=50
bpm = 60
noiseSD=0.01
samplingRate=50
trainingSignalLength=5 #in seconds
predictionSignalLength=5
internalStateSamples=int(samplingRate*trainingSignalLength)
predictSamples=int(samplingRate*predictionSignalLength)
showPlot=True

# Dataset construction
print "Creating data set..."
trainingSet = generateEcg(50,trainingSignalLength+predictionSignalLength)
testSet = generateEcg(1,trainingSignalLength+predictionSignalLength)
internalStateWindow = [xrange(i,i+windowSize) for i in xrange(internalStateSamples-windowSize)]
trainTestWindow= [xrange(i,i+windowSize) for i in xrange(internalStateSamples-windowSize,internalStateSamples+predictSamples-windowSize)]


# Network initialization
print "Initializing Network"
denseLayer=0
lstmLayer=1
network = Network(windowSize)
network.add_level(lstmLayer, 100)
network.add_level(denseLayer, 8,'tanh')
network.add_level(denseLayer, 1,'tanh') 

if showPlot:
    ecg.ecgInteractivePlotAxes(0,internalStateSamples+predictSamples,-1,1)


for epoch in range(maxEpochs):
    print("Epoch "+str(epoch)+"/"+str(maxEpochs))
    getAccuracy()
    train()
    if epoch%10==0:
       network.save_weights_and_biases("mnist_epoch_"+str(epoch))
  

