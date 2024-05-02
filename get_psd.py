'''This script estimates the PSD from data and returns interpolated PSD function.
It also defines a joint PSD for the two detectors.
'''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import tukey
from scipy.interpolate import interp1d


def get_psds(times, data_H1, data_L1, make_plots=False):

    # define FFT parameters
    dt = times[1] - times[0]
    fs = int(1./dt)
    NFFT = int(4 * fs)  # use 4 seconds of data for each FFT
    NOVL = int( 1 * NFFT / 2 )  # number of points of overlap between segments used in Welch averaging
    psd_window = tukey(NFFT, alpha=1./4)
    
    NFFTH = int(1 * fs)  # use 4 seconds of data for each FFT
    NOVLH = int(1 * NFFTH / 2)  # number of points of overlap between segments used in Welch averaging
    psd_windowH = tukey(NFFTH, alpha=1./4)

    # get PSDs
    # Pxx_H1, freqs = mlab.psd(data_H1, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
    Pxx_H1, freqsH = mlab.psd(data_H1, Fs=fs, NFFT=NFFTH, window=psd_windowH, noverlap=NOVLH)
    Pxx_L1, freqsL = mlab.psd(data_L1, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
    fminH = min(freqsH)
    fmaxH = max(freqsH)
    fminL = min(freqsL)
    fmaxL = max(freqsL)
    
    # interpolations of the PSDs computed above for whitening
    psd_H1 = interp1d(freqsH, Pxx_H1)
    psd_L1 = interp1d(freqsL, Pxx_L1)
    
    # plot PSDs
    if make_plots:
        plt.subplot(1, 2, 1)
        plt.loglog(freqsH, Pxx_H1, label='Hanford PSD estimate')
        freqs_interpH = np.linspace(fminH, fmaxH, 1000)
        plt.loglog(freqs_interpH, psd_H1(freqs_interpH), label='Hanford interpolation')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend(loc='lower left')
        plt.xlim(10, 3000)
        plt.subplot(1, 2, 2)
        plt.loglog(freqsL, Pxx_L1, label='Livingston PSD estimate')
        freqs_interpL = np.linspace(fminL, fmaxL, 1000)
        plt.loglog(freqs_interpL, psd_L1(freqs_interpL), label='Livingston interpolation')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('PSD')
        plt.legend(loc='lower left')
        plt.xlim(10, 3000)
        plt.show()

    return [psd_H1, psd_L1]


# get individual PSDs
def individual_psds():
    times_psd = np.loadtxt('data/times_psd.dat')
    H1_data_psd = np.loadtxt('data/H1_psd.dat')
    L1_data_psd = np.loadtxt('data/L1_psd.dat')
    psd_H1, psd_L1 = get_psds(times_psd, H1_data_psd, L1_data_psd)
    return [psd_H1, psd_L1]


# joint PSD between Hanford and Livingston
def joint_psd(freqs):
    times_psd = np.loadtxt('data/times_psd.dat')
    H1_data_psd = np.loadtxt('data/H1_psd.dat')
    L1_data_psd = np.loadtxt('data/L1_psd.dat')
    psd_H1, psd_L1 = get_psds(times_psd, H1_data_psd, L1_data_psd)
    return (1/psd_H1(freqs) + 1/psd_L1(freqs))**(-1)




# times_psd = np.loadtxt('data/times_psd.dat')
# H1_data_psd = np.loadtxt('data/H1_psd.dat')
# L1_data_psd = np.loadtxt('data/L1_psd.dat')
# get_psds(times_psd, H1_data_psd, L1_data_psd, make_plots=True)

