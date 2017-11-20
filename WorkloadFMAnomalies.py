'''
Workload which is a string of characters from a file
'''

import numpy as np
import scipy
import random
import collections
import time
import math
from FixedPoint import FixedPoint
from Settings import *

# Hyper-parameters
DATASET_FILE = "fm_data.bin"
NUM_TRAINING_EXAMPLES = 100000
TRAIN_SET_RATIO = 0.8
TEST_SET_RATIO = 0.1
GAUSSIAN_NOISE_SD = 0.02

def load_real_data():
    x = scipy.fromfile(open(DATASET_FILE), dtype=scipy.complex64)
    i_s = np.real(x[:NUM_TRAINING_EXAMPLES])
    q_s = np.imag(x[:NUM_TRAINING_EXAMPLES])
    return i_s,q_s

def make_data(data, window_size):
    X = []
    win_size = int(window_size)
    for i in range(len(data)-win_size):
        win = data[i:i+win_size]
        X.append(win)
    return np.array(X)

def make_fft_windows(i,q,window_size):
    sig = i + q*1j;
    sig_window = make_data(sig, window_size)
    data = []
    for sig_w in sig_window:
        fft_res = np.fft.fft(sig_w)
        fft_r = np.real(fft_res)
        fft_i = np.imag(fft_res)
        data.append(np.concatenate((fft_r,fft_i)).tolist())
    return data



def convert_to_list(c):
    l = [[x] for x in c]
    return l

def WorkloadFMAnomalies(window_size):
    i_s,q_s = load_real_data();
    data = make_fft_windows(i_s,q_s,int(window_size/2))

    training_set_x = []
    training_set_y = []
    test_set_x = []
    test_set_y = []
    validation_set_x = []
    validation_set_y = []

    train_size = int(len(data)*TRAIN_SET_RATIO)
    test_size = int(len(data)*TEST_SET_RATIO)

    count = 0
    for i in range(0,train_size):
        training_set_x.append(convert_to_list(data[i]))
        training_set_y.append(0)
    for i in range(train_size,train_size+test_size):
        if (random.randint(1,8) == 1):
            y = 1
            noise = data[i] + np.random.normal(0,GAUSSIAN_NOISE_SD,32) # add some gaussian noise
        else:
            noise = data[i]
            y = 0
        test_set_x.append(convert_to_list(noise))
        test_set_y.append(y)
    for i in range(train_size+test_size,len(data)):
        if (random.randint(1,8) == 1):
            y = 1
            noise = data[i] + np.random.normal(0,GAUSSIAN_NOISE_SD,32) # add some gaussian noise
        else:
            noise = data[i]
            y = 0
        validation_set_x.append(convert_to_list(noise))
        validation_set_y.append(y)



    

    training_set = zip(training_set_x, training_set_y)
    validation_set = zip(validation_set_x, validation_set_y)
    test_set = zip(test_set_x, test_set_y)

    return((training_set, validation_set, test_set))
