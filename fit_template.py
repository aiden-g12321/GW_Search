import numpy as np
import matplotlib.pyplot as plt
from waveform_tools import *
from signal_tools import *



# fit whitened template to whitened data
def fit_template(strain_data, times, psd_inter, max_params):
    
    # get the Fourier frequencies of data
    n = len(strain_data)
    dt = times[1] - times[0]
    f_sample = int(1./dt)
    freqs = np.fft.fftfreq(n) * f_sample
    df = freqs[1] - freqs[0]
    
    # fft data
    dwindow = tukey(n, alpha=1./4)
    data_fft = np.fft.fft(strain_data*dwindow) / f_sample
    
    # get whitened / bandpassed data
    whiten_bp_data = get_white_bp(times, strain_data, psd_inter)

    # get template in frequency-domain (in-phase and quadrature-phase)
    waveform_freq = np.zeros(n, dtype='complex')
    quad_phase = np.zeros(n, dtype='complex')
    keep_indices = np.where(np.bitwise_and(freqs > 35., freqs < 350.))
    waveform_freq[keep_indices] = get_waveform_freq(freqs[keep_indices], max_params)
    quad_params = np.array(max_params).copy()
    quad_params[3] = np.pi / 2.
    quad_phase[keep_indices] = get_waveform_freq(freqs[keep_indices], quad_params)
    
    # get psd
    psd = psd_inter(np.abs(freqs))
    
    # normalize templates
    sigma_sq = 2 * (waveform_freq * waveform_freq.conjugate() / psd).sum() * df
    normal_waveform = waveform_freq / np.sqrt(sigma_sq)
    normal_quad = quad_phase / np.sqrt(sigma_sq)
    
    # find phase that minimizes chi-squared with data
    data_with_quad = np.real(2 * (data_fft * normal_quad.conjugate() / psd).sum() * df)
    data_with_inphase = np.real(2 * (data_fft * normal_waveform.conjugate() / psd).sum() * df)
    phi = np.arctan2(data_with_quad, data_with_inphase)
    
    # find phase that minimizes chi-squared
    template_freq = np.zeros(n, dtype='complex')
    phase_max_params = np.array(max_params).copy()
    phase_max_params[3] = phi
    template_freq[keep_indices] = get_waveform_freq(freqs[keep_indices], phase_max_params)
    
    # whiten template
    template_freq = template_freq / np.sqrt(psd)
    
    # inverse Fourier transform to time-domain
    template_time = np.real(np.fft.ifft(template_freq) * f_sample)

    # normalize to fit whitened / bandpassed data
    template_time *= max(whiten_bp_data) / max(template_time)
    
    # find time that maximizes inner product
    index_window = 500
    peak_data_index = list(whiten_bp_data).index(max(whiten_bp_data))
    peak_template_index = list(template_time).index(max(template_time))
    index_shift = peak_data_index - peak_template_index
    SNRsqs = []
    for i in range(-index_window, index_window):
        template_shifted = np.roll(template_time, int(index_shift + i))
        SNRsq = sum(template_shifted * whiten_bp_data) * dt
        SNRsqs.append(SNRsq)
    max_index = SNRsqs.index(max(SNRsqs)) - index_window
    template_time = np.roll(template_time, int(index_shift + max_index))
    
    return template_time



# plot templates over whitened data
def plot_templates(max_params, H_data, L_data):
    
    # get PSDs
    H1_psd, L1_psd = individual_psds()
    
    # get whitened / bandpassed strain data
    H_white_bp = get_white_bp(times, H_data, H1_psd)
    L_white_bp = get_white_bp(times, L_data, L1_psd)

    # get templates maximized over time and phase
    H_template_time = fit_template(H, times, H1_psd, max_params)
    L_template_time = fit_template(L, times, L1_psd, max_params)
    
    plt.subplot(2, 1, 1)
    plt.plot(times, H_white_bp, label='Hanford')
    plt.plot(times, H_template_time, label='template')
    plt.legend(loc='upper left')
    plt.xlabel('time (s)')
    plt.ylabel('strain')
    plt.subplot(2, 1, 2)
    plt.plot(times, L_white_bp, label='Livingston')
    plt.plot(times, L_template_time, label='template')
    plt.legend(loc='upper left')
    plt.xlabel('time (s)')
    plt.ylabel('strain')
    plt.show()
    
    return




max_params = [1.87201150e-04, 1.77350168e-04, 0.00000000e+00, 0.00000000e+00, 1.02927125e+16]
max_SNR_segment_index = 8
times = np.loadtxt('data/times_' + str(max_SNR_segment_index) + '.dat')
H = np.loadtxt('data/H1_' + str(max_SNR_segment_index) + '.dat')
L = np.loadtxt('data/L1_' + str(max_SNR_segment_index) + '.dat')
plot_templates(max_params, H, L)








