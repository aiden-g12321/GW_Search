'''This script stores whitening and bandpassing methods.
'''



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal.windows import tukey
from constants import *
from get_psd import *



# whiten strain data
def whiten(strain, interp_psd, dt, phase_shift=0, time_shift=0):
    Nt = len(strain)
    # take the fourier transform of the data
    freqs = np.fft.rfftfreq(Nt, dt)

    # whitening: transform to freq domain, divide by square root of psd, then
    # transform back, taking care to get normalization right.
    hf = np.fft.rfft(strain)
    
    # apply time and phase shift
    hf = hf * np.exp(-1.j * 2 * np.pi * time_shift * freqs - 1.j * phase_shift)
    norm = 1./np.sqrt(1./(dt*2))
    white_hf = hf / np.sqrt(interp_psd(freqs)) * norm
    white_ht = np.fft.irfft(white_hf, n=Nt)
    return white_ht


# bandpass strain data
def bandpass(strain, fband, fs):
    bb, ab = butter(4, [fband[0]*2./fs, fband[1]*2./fs], btype='band')
    normalization = np.sqrt((fband[1]-fband[0])/(fs/2))
    strain_bp = filtfilt(bb, ab, strain) / normalization
    return strain_bp


# plot whiten and bandpassed strain data
def plot_strain_data(times, strain_data, psd_inter):
    
    fband = [35.0, 350.0]
    dt = times[1] - times[0]
    fs = int(1./dt)
    
    # create our 8 second data window
    window_len = int(time_segment * fs)
    dwindow = tukey(window_len, alpha=1./4)

    # plot original strain data
    plt.figure(figsize=(8, 8))
    plt.subplot(4, 1, 1)
    indxt = range(len(times))
    plt.plot(times[indxt], strain_data[indxt], 'blue', 
             label='H/L Data', linewidth=.5)
    plt.legend()

    # plot windowed data
    plt.subplot(4, 1, 2)
    strain_windowed = dwindow * strain_data[indxt]
    # strain_windowed = dwindow * H_data
    plt.plot(times[indxt], strain_windowed, 'green', 
             label='Windowed Data', linewidth=.5)
    plt.legend()
    
    
    # plot whitened data
    plt.subplot(4, 1, 3)
    strain_whitened = whiten(strain_windowed, psd_inter, dt)
    plt.plot(times[indxt], strain_whitened, 'red', 
             label='Whitened Data', linewidth=.5)
    plt.legend()

    # plot bandpassed data
    plt.subplot(4, 1, 4)
    strain_bp = bandpass(strain_whitened, fband, fs)
    plt.plot(times[indxt], strain_bp, 'black', 
             label='Bandpassed Data', linewidth=.5)

    plt.legend()
    plt.yticks([-6, -3, 0, 3, 6, 9])
    plt.tight_layout()
    plt.ylim([-8, 8])
    plt.show()
    
    return
    

# get whitened / bandpassed data array
def get_white_bp(times, strain_data, psd_inter):
    
    # frequency parameters
    fband = [35.0, 350.0]
    dt = times[1] - times[0]
    fs = int(1./dt)
    
    # window data
    window_len = int(time_segment * fs)
    dwindow = tukey(window_len, alpha=1./4)
    strain_windowed = dwindow * strain_data
    # whiten data
    strain_whitened = whiten(strain_windowed, psd_inter, dt)
    # bandpass data
    strain_bp = bandpass(strain_whitened, fband, fs)

    return strain_bp

