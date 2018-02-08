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
    ecg = np.tile(ecg , int(timeLength * bpm/60.)+1)

    # Trim it to the right size
    ecg=ecg[0:int(30*timeLength * bpm/60.)]

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
plt.subplot(211)
axes = plt.gca()
axes.set_xlim(0, 1500)
axes.set_ylim(-1, +1)
line1, = axes.plot([], [],label='Test signal')
line2, = axes.plot([], [], '--',label='Predicted Signal')
plt.legend(loc='upper left')
plt.xticks([])
plt.subplot(212)
axes2 = plt.gca()
axes2.set_xlim(0, 1500)
axes2.set_ylim(-1, +1)
line1_2, = axes2.plot([], [])
line2_2, = axes2.plot([], [], '--')
plt.subplots_adjust(left=None, bottom=0.170, right=None, top=None,
            wspace=None, hspace=0.0)
plt.text(-60, .8, 'Amplitude (V)', ha='center', va='center', rotation='vertical')


def ecgInteractivePlotAxes(x_init,x_end,y_init,y_end):
    axes.set_xlim(x_init, x_end)
    axes.set_ylim(y_init, y_end)
    axes2.set_xlim(x_init, x_end)
    axes2.set_ylim(y_init, y_end)
    axes2.set_xlabel("Samples")

# Plot ECG signal and predicted signal
def ecgInteractivePlot(ecg,predicted,ecg2,predicted2,samplingRate):
    line1.set_xdata(xrange(len(ecg)))
    line1.set_ydata(ecg)
    line2.set_xdata(xrange(len(ecg),len(ecg)+len(predicted)))
    line2.set_ydata(predicted)

    line1_2.set_xdata(xrange(len(ecg2)))
    line1_2.set_ydata(ecg2)
    line2_2.set_xdata(xrange(len(ecg2),len(ecg2)+len(predicted2)))
    line2_2.set_ydata(predicted2)
    plt.draw()
    plt.pause(1e-17)
