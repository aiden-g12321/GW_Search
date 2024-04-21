import numpy as np
import matplotlib.pyplot as plt
from constants import *
from waveform_tools import *
from template_bank import *
from get_psd import *


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
    
    # get template in frequency-domain
    waveform_freq = get_waveform_freq(fs, template_params)
    template_SNRsq = inner(waveform_freq, waveform_freq, psd, df)
    template_SNR = np.sqrt(np.abs(template_SNRsq))
    
    # compute z-statistic in frequency-domain
    optimal = 4 * data_freq * waveform_freq.conjugate() / psd
    
    # add negative and zero frequencies for ifft
    optimal_with_negs = np.zeros(n, dtype='complex')
    optimal_with_negs[1:int(n/2)] = optimal
    
    # get SNR time series with inverse Fourier transform
    optimal_time = np.fft.ifft(optimal_with_negs) * f_sample
    SNR_series = np.abs(optimal_time / template_SNR)   
    
    return SNR_series











#################################################################
######################### TESTING ###############################
#################################################################


times = np.loadtxt('data/times_0.dat')
H1 = np.loadtxt('data/H1_0.dat')
L1 = np.loadtxt('data/L1_0.dat')

params = np.array([m1_measured_sec, m2_measured_sec, 0., 0., Dl100Mpc])


get_SNR_series(params, H1, times)

