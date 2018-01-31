'''
An example on how save and load weights
'''

import sys
sys.path.append('..')
from FixedPoint import FixedPoint
from Lstm import LSTM
from Dense import Dense
from Settings import *
from Network import *

# Network has 2 dense layers with 3 inputs of any dimension
print "Initializing Network"
num_inputs=3
denseLayer=0
lstmLayer=1

# Network 1 - Is created with random weights and bias
network1 = Network(num_inputs)
network1.add_level(lstmLayer, 2)
network1.add_level(denseLayer, 1)

# Network 2 - Is created with random weights and bias
network2 = Network(num_inputs)
network2.add_level(lstmLayer, 2)
network2.add_level(denseLayer, 1)

# If you are using fixed point, this will make sure that weights are different
network2.level[1].B=-1.23456

# Create an arbitraty input to test the networks
myInput=[[1],[1],[1]]
currentEpoch=0

# This will show that the two networks will have different outputs (since weights are different)
print("\nOutput for network 1 is")
print(network1.forward_pass(myInput))
print("\nOutput for network 2 is")
print(network2.forward_pass(myInput))

# We now "transfer" the weights from Network1 to Network2
print("\nLoading network1's weights and biases on network2:")
network1.save_weights_and_biases("Epoch_"+str(currentEpoch))
network2.load_weights_and_biases("Epoch_"+str(currentEpoch))
network2.level[0].reset()

# And Network2 should be able to have the same results as Network1
print("\nOutput for network 2 is")
print(network2.forward_pass(myInput))
