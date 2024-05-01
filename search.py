import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import tukey

from constants import *
from waveform_tools import *
from template_bank import *
from get_psd import *
from signal_tools import *


# get SNR time series given data and parameters for template
def get_SNR_series(template_params, strain_data, times, psd_inter):
    
    n = len(strain_data)
    dt = times[1] - times[0]
    f_sample = int(1./dt)
    
    # get the Fourier frequencies of data
    freqs = np.fft.fftfreq(n) * f_sample
    df = freqs[1] - freqs[0]
    
    # bandpass data
    # strain_data = bandpass(strain_data, [35.0, 350.0], f_sample)
    
    # tukey window for taking the fft of our template and data
    dwindow = tukey(n, alpha=1./4)
    
    # compute the template and data ffts
    template_fft_positives = get_waveform_freq(freqs[1:n//2], template_params)
    data_fft = np.fft.fft(strain_data*dwindow) / f_sample
    template_fft = np.zeros(len(data_fft), dtype='complex')
    template_fft[1:n//2] = template_fft_positives
    template_fft[n//2+1:] = template_fft_positives.conjugate()

    # -- Zero out negative frequencies
    negindx = np.where(freqs < 0)
    data_fft[negindx] = 0

    # get PSD at frequencies
    psd = psd_inter(np.abs(freqs))

    optimal = data_fft * template_fft.conjugate() / psd  
    optimal_time = 4 * np.fft.ifft(optimal) * f_sample
    
    sigmasq = 2 * (template_fft * template_fft.conjugate() / psd).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time / sigma
    
    peaksample = int(n / 2)  # location of peak in the template
    # SNR_complex = np.roll(SNR_complex,peaksample)
    SNR = abs(SNR_complex)
    
    return SNR


# get maximized SNR time series
def get_max_SNR_series():
    
    # store template bank
    bank_freqs = np.linspace(20., 2048., 2**12+1)
    bank_psd = joint_psd(bank_freqs)
    bank_df = bank_freqs[1] - bank_freqs[0]
    paramss, metrics = get_template_bank(bank_freqs, bank_psd, bank_df)
    num_templates = len(paramss)
    
    # initialize SNR time series
    H1_series = []
    L1_series = []
    
    for i in range(num_segments):
        
        # load data and times for segment
        times = np.loadtxt('data/times_' + str(i) + '.dat')
        H1 = np.loadtxt('data/H1_' + str(i) + '.dat')
        L1 = np.loadtxt('data/L1_' + str(i) + '.dat')
        
        # array to store series for each template
        H1_array = np.zeros((num_templates, len(times)))
        L1_array = np.zeros((num_templates, len(times)))
        
        # get SNR series for each template in each segment
        for j in range(num_templates):
            H1_array[j] = get_SNR_series(paramss[j], H1, times)
            L1_array[j] = get_SNR_series(paramss[j], L1, times)
        
        # get maximized SNR series over templates
        H1_series.extend(np.max(H1_array, axis=0))
        L1_series.extend(np.max(L1_array, axis=0))
        
    return [H1_series, L1_series]



#################################################################
######################### TESTING ###############################
#################################################################


times = np.loadtxt('data/times_event.dat')
H1 = np.loadtxt('data/H1_event.dat')
L1 = np.loadtxt('data/L1_event.dat')

plt.plot(times, H1, label='Hanford')
plt.plot(times, L1, label='Livingston')
plt.legend()
plt.show()

params = [m1_measured_sec, m2_measured_sec, 0., 0., Dl100Mpc]
H_psd, L_psd = individual_psds()
H1_SNR_series = get_SNR_series(params, H1, times, H_psd)
L1_SNR_series = get_SNR_series(params, L1, times, L_psd)

fs = np.linspace(0.1, 2048., 2**12+1)
plt.subplot(3, 1, 1)
plt.loglog(fs, H_psd(fs), label='Hanford')
plt.loglog(fs, L_psd(fs), label='Livingston')
plt.legend(loc='upper right')
plt.subplot(3, 1, 2)
plt.plot(times, H1_SNR_series, label='Hanford')
plt.legend(loc='upper left')
plt.subplot(3, 1, 3)
plt.plot(times, L1_SNR_series, label='Livingston', color='orange')
plt.legend(loc='upper left')
plt.show()
