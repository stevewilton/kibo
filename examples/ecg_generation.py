import matplotlib.pyplot as plt
import numpy as np 
import scipy.signal as sig
import time

def generateEcg(bpm, timeLength, noiseSD=0.01, samplingRate=200 ):

    # Single Daubechies wavelet 
    ecg = sig.wavelets.daub(10)

    # Add pause after each heart beat
    ecg = np.concatenate([ecg,np.zeros(10)])

    # Concatenate multiple heartbeats
    ecg = np.tile(ecg , int(timeLength * bpm/60.))

    # Add first level of gaussian noise 
    ecg = np.random.normal(0, noiseSD, len(ecg)) + ecg

    # Resampling
    ecg = sig.resample(ecg, int(samplingRate * timeLength))

    # Add second level of gaussian noise 
    ecg = np.random.normal(0, noiseSD, len(ecg)) + ecg

    return ecg

# Plot ECG signal
def plotEcg(ecg,samplingRate,bpm):
    plt.plot(ecg)
    plt.xlabel("Time in s")
    plt.ylabel("Signal Amplitude (V)")
    plt.title("Simulated ECG at "+str(samplingRate)+" Hz and "+str(bpm)+" bpm")
    plt.show()

# Plot ECG signal and predicted signal
def plotEcgPredicted(ecg,predicted,samplingRate,bpm):
    plt.plot(ecg)
    plt.plot(xrange(len(ecg),len(ecg)+len(predicted)),predicted,'--')
    plt.xlabel("Time in s")
    plt.ylabel("Signal Amplitude (V)")
    plt.title("Predictions for simulated ECG at "+str(samplingRate)+" Hz and "+str(bpm)+" bpm")
    plt.show()



plt.show()
axes = plt.gca()
axes.set_xlim(0, 1500)
axes.set_ylim(-1, +1)
line1, = axes.plot([], [])
line2, = axes.plot([], [], '--')

def ecgInteractivePlotAxes(x_init,x_end,y_init,y_end):
    axes.set_xlim(x_init, x_end)
    axes.set_ylim(y_init, y_end)

# Plot ECG signal and predicted signal
def ecgInteractivePlot(ecg,predicted,samplingRate,bpm):
    print("updating plot")
    line1.set_xdata(xrange(len(ecg)))
    line1.set_ydata(ecg)
    line2.set_xdata(xrange(len(ecg),len(ecg)+len(predicted)))
    line2.set_ydata(predicted)
    plt.draw()
    plt.pause(1e-17)
    plt.xlabel("Time in s")
    plt.ylabel("Signal Amplitude (V)")
    plt.title("Predictions for simulated ECG at "+str(samplingRate)+" Hz and "+str(bpm)+" bpm")