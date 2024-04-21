'''This script contains methods for waveform generation, operations (such as inner product,
template metric, and mismatch), and inverse FFTs'''


import numpy as np
import matplotlib.pyplot as plt
from PhenomA import *
from constants import *
from get_psd import *
from scipy.integrate import romb as integrate
from scipy.signal import tukey


##################################################################
################### WAVEFORM GENERATION ##########################
##################################################################


# function to change mass coordinates
def mass_transform(comp_masses):
    m1, m2 = comp_masses
    M = m1 + m2
    eta = m1 * m2 / M**2
    return np.array([M, eta])


# modify PhenomA methods to input parameters [m1, m2, t0, phi0, Dl]
# all units are seconds

# get amplitude of signal in frequency-domain
def get_amp(fs, params):
    m1, m2, t0, phi0, Dl = params
    M, eta = mass_transform([m1, m2])
    return Aeff(fs, M, eta, Dl=Dl)

# get phase of signal in frequency-domain
def get_phase(fs, params):
    m1, m2, t0, phi0, Dl = params
    M, eta = mass_transform([m1, m2])
    return Psieff(fs, M, eta, t0=t0, phi0=phi0)


# get frequency-domain waveform
def get_waveform_freq(fs, params):
    return get_amp(fs, params) * np.exp(1.j * get_phase(fs, params))



##################################################################
################### WAVEFORM OPERATIONS ##########################
##################################################################



# define inner product between waveforms in frequency-domain
# uses scipy's rhombus numerical integration (requires 2^n+1 frequencies)
def inner(a, b, Ss, df):
    integrand = 4. * (np.real(a)*np.real(b) + np.imag(a)*np.imag(b)) / Ss
    inner_prod = integrate(integrand, dx=df)
    return inner_prod


step_sizes = [1.e-10, 1.e-10, 1.e-5, 1.e-5, 1.e-5]
# calculate partial derivative of frequency-domain waveform
def partial_waveform(freqs, params, index):
    dstep = np.zeros(num_params)
    dstep[index] = step_sizes[index]
    waveform1 = get_waveform_freq(freqs, params - dstep)
    waveform2 = get_waveform_freq(freqs, params + dstep)
    return (waveform2 - waveform1) / (2 * step_sizes[index])


# calculate match between waveforms
def match(params1, params2, freqs, Ss, df):
    waveform1 = get_waveform_freq(freqs, params1)
    waveform2 = get_waveform_freq(freqs, params2)
    SNRsq1 = inner(waveform1, waveform1, Ss, df)
    SNRsq2 = inner(waveform2, waveform2, Ss, df)
    cross = inner(waveform1, waveform2, Ss, df)
    return cross / np.sqrt(SNRsq1 * SNRsq2)


# get components of metric on template space
def metric(freqs, params, Ss, df):
    waveform = get_waveform_freq(freqs, params)
    normalization = 1 / np.sqrt(inner(waveform, waveform, Ss, df))
    norm_waveform = normalization * waveform
    partials = [partial_waveform(freqs, params, i) for i in range(num_params)]
    norm_partials = normalization * np.array(partials)
    SNRsq = inner(norm_waveform, norm_waveform, Ss, df)
    SNR = np.sqrt(SNRsq)
    metric_comp = np.zeros((num_params, num_params))
    for i in range(num_params):
        for j in range(i, num_params):
            first_term = inner(norm_partials[i], norm_partials[j], Ss, df) / SNR
            second_term = inner(norm_waveform, norm_partials[i], Ss, df) * inner(norm_waveform, norm_partials[j], Ss, df) / SNRsq
            metric_comp[i,j] = metric_comp[j,i] = first_term - second_term
    return metric_comp


# project reference time and phase out of metric
def projected_metric(freqs, params, Ss, df):
    metric_comp = metric(freqs, params, Ss, df)
    first_proj = np.zeros((num_params, num_params))
    second_proj = np.zeros((num_params, num_params))
    for i in range(num_params):
        for j in range(i, num_params):
            first_proj[i,j] = first_proj[j,i] = metric_comp[i,j] - metric_comp[i,2]*metric_comp[j,2]/metric_comp[2,2]
    for i in range(num_params):
        for j in range(i, num_params):
            second_proj[i,j] = second_proj[j,i] = first_proj[i,j] - first_proj[i,3]*first_proj[j,3]/first_proj[3,3]
    return second_proj


# compute mismatch between waveforms
def get_mismatch(proj_metric, params1, params2):
    delta = np.array(params2) - np.array(params1)
    mismatch = 0
    for i in range(2):
        for j in range(2):
            mismatch += (1/2) * proj_metric[i,j] * delta[i] * delta[j]
    return mismatch


# input: parameter list in seconds or [m1, m2] in seconds
# returns: [m1, m2] in solar masses
def convert_solar(params):
    params = np.array(params)
    return params[:2] / MTSUN_SI



##################################################################
################### FOURIER TRANSFORM ############################
##################################################################


# Fourier transform data
def fft(times, data_time):
    n = len(data_time)
    dt = times[1] - times[0]
    f_sample = int(1./dt)
    # get the Fourier frequencies of data
    freqs = np.fft.fftfreq(n)*f_sample
    # for taking the fft of our template and data
    dwindow = tukey(n, alpha=1./4)
    # compute the data fft
    data_freq = np.fft.fft(data_time*dwindow) / f_sample
    # freqs = freqs[:int(n/2)]  # remove negative frequencies
    # data_freq = data_freq[:int(n/2)]  # remove negative frequenies
    return [freqs, data_freq]


# inverse Fourier transform
def ifft(waveform_freq, freqs):
    Nf = len(freqs)
    df = freqs[1] - freqs[0]
    Nt = 2 * Nf - 1
    dt = 1 / (Nt * df)
    T = Nt * dt
    waveform_time = np.real(np.fft.irfft(waveform_freq, n=Nt))

    # roll time-domain waveform array to place merger later
    index_merger = list(waveform_time).index(max(waveform_time))
    merger_placement = 5./6.  # fraction of window where to place merger
    index_shift = round(Nt * merger_placement) - index_merger
    waveform_shifted = np.roll(waveform_time, index_shift)
    times_shifted = np.linspace(-merger_placement * T, (1-merger_placement)*T, Nt)
    
    return [times_shifted, waveform_shifted]



##################################################################
######################### TESTING ################################
##################################################################


times = np.loadtxt('data/times_0.dat')
H1 = np.loadtxt('data/H1_0.dat')
L1 = np.loadtxt('data/L1_0.dat')


fs, data_freq = fft(times, H1)





