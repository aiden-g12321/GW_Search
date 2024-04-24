import numpy as np
import matplotlib.pyplot as plt
from constants import *
from waveform_tools import *
from template_bank import *
from get_psd import *
from scipy.signal import tukey


# get SNR time series given data and parameters for template
def get_SNR_series(template_params, data_time, times):
    
    # fft data and cut out non-positive frequencies
    fs, data_freq = fft(times, data_time)
    n = len(fs)
    fs = fs[1:int(n/2)]
    dt = times[1] - times[0]
    f_sample = int(1./dt)
    df = fs[1] - fs[0]
    data_freq = data_freq[1:int(n/2)]
    
    # get psd
    psd = joint_psd(fs)
    
    # get normalized template in frequency-domain
    template_freq = get_waveform_freq(fs, template_params)
    template_SNRsq = 2 * (template_freq * template_freq.conjugate() / psd).sum() * df
    template_SNR = np.sqrt(np.abs(template_SNRsq))
    template_freq = template_freq / template_SNR
    
    # compute z-statistic in frequency-domain
    optimal = 4 * data_freq * template_freq.conjugate() / psd
    
    # add negative and zero frequencies for ifft
    optimal_with_negs = np.zeros(n, dtype='complex')
    optimal_with_negs[1:int(n/2)] = optimal
    
    # get SNR time series with inverse Fourier transform
    optimal_time = np.fft.ifft(optimal_with_negs) * f_sample
    SNR_series = np.abs(optimal_time / template_SNR)   
    
    return SNR_series


# do search
def search():
    
    # store template bank
    bank_freqs = np.linspace(20., 2048., 2**12+1)
    bank_psd = joint_psd(bank_freqs)
    bank_df = bank_freqs[1] - bank_freqs[0]
    paramss, metrics = get_template_bank(bank_freqs, bank_psd, bank_df)
    num_templates = len(paramss)
    
    for i in range(num_segments):
        
        # load data and times for segment
        times = np.loadtxt('data/times_' + str(i) + '.dat')
        H1 = np.loadtxt('data/H1_' + str(i) + '.dat')
        L1 = np.loadtxt('data/L1_' + str(i) + '.dat')
        
        
        max_SNRs_H1 = np.zeros(num_templates)
        max_SNRs_L1 = np.zeros(num_templates)
        for j in range(num_templates):
            SNR_series_H1 = get_SNR_series(paramss[j], H1, times)
            SNR_series_L1 = get_SNR_series(paramss[j], L1, times)
            max_SNRs_H1[j] = max(SNR_series_H1)
            max_SNRs_L1[j] = max(SNR_series_L1)
        
        print(max(max_SNRs_H1))
        print(max(max_SNRs_L1))
        print()
        
        
    return



#################################################################
######################### TESTING ###############################
#################################################################


times = np.loadtxt('data/times_total.dat')
H1 = np.loadtxt('data/H1_total.dat')

params = np.array([m1_measured_sec, m2_measured_sec, 0., 0., Dl100Mpc])

SNR_series = get_SNR_series(params, H1, times)

plt.plot(times, SNR_series)
plt.show()
