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

    # get PSDs
    Pxx_H1, freqs = mlab.psd(data_H1, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
    Pxx_L1, freqs = mlab.psd(data_L1, Fs=fs, NFFT=NFFT, window=psd_window, noverlap=NOVL)
    fmin = min(freqs)
    fmax = max(freqs)
    
    # interpolations of the PSDs computed above for whitening
    psd_H1 = interp1d(freqs, Pxx_H1)
    psd_L1 = interp1d(freqs, Pxx_L1)
    
    # save frequencies and PSDs
    np.savetxt('frequencies.txt', freqs)
    np.savetxt('psd_H1.txt', Pxx_H1)
    np.savetxt('psd_L1.txt', Pxx_L1)
    
    # plot PSDs
    if make_plots:
        plt.loglog(freqs, Pxx_H1, label='PSD estimate')
        freqs_interp = np.linspace(fmin, fmax, 1000)
        plt.loglog(freqs_interp, psd_H1(freqs_interp), label='interpolation')
        plt.xlabel('frequency (Hz)')
        plt.ylabel('Sn(t)')
        plt.legend(loc='upper right')
        plt.show()

    return [psd_H1, psd_L1]


# joint PSD between Hanford and Livingston
def joint_psd(times, data_H1, data_L1, freqs):
    psd_H1, psd_L1 = get_psds(times, data_H1, data_L1)
    return (1/psd_H1(freqs) + 1/psd_L1(freqs))**(-1)

