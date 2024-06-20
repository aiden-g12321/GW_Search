'''This script computes the SNR time-series given a template bank and strain data.
It contains methods to compute the SNR series for a given array of data, for full 2 minutes of data,
or the series that is maximized over the template bank.
'''


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
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
    # strain_data = bandpass(strain_data, [35.0, 1024.0], f_sample)
    
    # tukey window for taking the fft of our template and data
    dwindow = tukey(n, alpha=1./4)
    
    # compute the template and data ffts
    template_fft_positives = get_waveform_freq(freqs[1:n//2], template_params)
    data_fft = np.fft.fft(strain_data*dwindow)
    template_fft = np.zeros(len(data_fft), dtype='complex')
    template_fft[1:n//2] = template_fft_positives
    template_fft[n//2+1:] = template_fft_positives.conjugate()
    
    # apply time shift so iFFT is centered correctly
    tau = get_tau(template_params)
    template_fft *= np.exp(1.j * 2 * np.pi * freqs * tau)

    # -- Zero out negative frequencies
    negindx = np.where(freqs < 0)
    data_fft[negindx] = 0

    # get PSD at frequencies
    psd = psd_inter(np.abs(freqs))

    # compute SNR time-series (optimal statistic)
    optimal = data_fft * template_fft.conjugate() / psd
    optimal_time = 4 * np.fft.ifft(optimal) * f_sample
    
    # normalize with template SNR
    sigmasq = 2 * (template_fft * template_fft.conjugate() / psd).sum() * df
    sigma = np.sqrt(np.abs(sigmasq))
    SNR_complex = optimal_time / sigma
    SNR = abs(SNR_complex)
    
    # cut out first and last 0.5 seconds
    first_time = times[0]
    last_time = times[-1]
    cut_time_low = first_time + time_overlap / 2
    cut_time_high = last_time - time_overlap / 2
    keep_indices = np.where(np.bitwise_and(times > cut_time_low, times < cut_time_high))
    
    times_keep = np.array(times)[keep_indices]
    SNR_keep = np.array(SNR)[keep_indices]
    
    return [times_keep, SNR_keep]



# get SNR time series for all 2 minutes of data
def get_full_SNR_series(template_params, H1_psd, L1_psd):
    
    # initialize time and SNR lists
    timessH = []
    SNRsH = []
    timessL = []
    SNRsL = []
    
    for i in range(num_segments):
        
        # load data and times for segment
        times = np.loadtxt('data/times_' + str(i) + '.dat')
        H1 = np.loadtxt('data/H1_' + str(i) + '.dat')
        L1 = np.loadtxt('data/L1_' + str(i) + '.dat')
        
        # get SNR time series segment
        timesH, SNR_H = get_SNR_series(template_params, H1, times, H1_psd)
        timesL, SNR_L = get_SNR_series(template_params, L1, times, L1_psd)
        
        # store on lists
        timessH.append(timesH)
        timessL.append(timesL)
        SNRsH.append(SNR_H)
        SNRsL.append(SNR_L)
        
    # flatten lists
    H_times = np.array(timessH).flatten()
    H_SNR = np.array(SNRsH).flatten()
    L_times = np.array(timessL).flatten()
    L_SNR = np.array(SNRsL).flatten()
    H1_series = np.array([H_times, H_SNR])
    L1_series = np.array([L_times, L_SNR])
    
    return [H1_series, L1_series]



# find template that maximizes SNR series
def max_template_SNR(H1_psd, L1_psd):
    
    # make template bank
    bank_freqs = np.linspace(20., 2048., 2**12+1)
    bank_psd = joint_psd(bank_freqs)
    bank_df = bank_freqs[1] - bank_freqs[0]
    paramss, metrics = get_template_bank(bank_freqs, bank_psd, bank_df)
    num_templates = len(paramss)
    
    # store SNR series (including times) over template bank
    # also store maximum SNR for each template
    H_SNR_series = []
    H_maxs = []
    L_SNR_series = []
    L_maxs = []
    
    for i in range(num_templates):
        print(i)
        params = paramss[i]
        H_series, L_series = get_full_SNR_series(params, H1_psd, L1_psd)
        H_maxs.append(max(H_series[1]))
        L_maxs.append(max(L_series[1]))
        H_SNR_series.append(H_series)
        L_SNR_series.append(L_series)
        
    # find template with maximum SNR
    H_max_index = H_maxs.index(max(H_maxs))
    L_max_index = L_maxs.index(max(L_maxs))
    H_max_params = paramss[H_max_index]
    L_max_params = paramss[L_max_index]
    H_SNR_series_max = H_SNR_series[H_max_index]
    L_SNR_series_max = L_SNR_series[L_max_index]
    
    return [H_max_params, L_max_params, H_SNR_series_max, L_SNR_series_max]


