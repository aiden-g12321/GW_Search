import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import romb as integrate
from constants import *
from get_psd import *
from IMRPhenomD import AmpPhaseFDWaveform, IMRPhenomDGenerateh22FDAmpPhase
import IMRPhenomD_const as imrc


# get h22 (frequency-domain) object
# h22 includes amplitude and phase as array
def get_h22(params, freqs):

    # reference frequency
    MfRef_in = 0.
    
    # unpack parameters
    # input mass in solar masses, luminosity distance in mega-parsecs
    m1, m2, t0, phi0, Dl = params
    chi1 = 0.
    chi2 = 0.
    # convert masses to kg
    m1_SI =  m1*imrc.MSUN_SI
    m2_SI =  m2*imrc.MSUN_SI
    # convert luminosity distance to meters
    Dl_SI = Dl * 1.e6 * imrc.PC_SI

    # initialize amplitude, phase, times, and time derivative
    num_freqs = len(freqs)
    amp_imr = np.zeros(num_freqs)
    phase_imr = np.zeros(num_freqs)
    time_imr = np.zeros(num_freqs)
    timep_imr = np.zeros(num_freqs)

    #the first evaluation of the amplitudes and phase will always be much slower, because it must compile everything
    h22 = AmpPhaseFDWaveform(num_freqs,freqs,amp_imr,phase_imr,time_imr,timep_imr,0.,t0)
    h22 = IMRPhenomDGenerateh22FDAmpPhase(h22,freqs,phi0,MfRef_in,m1_SI,m2_SI,chi1,chi2,Dl_SI)
    
    return h22


# get frequency-domain waveform for given parameters
def get_waveform_freq(params, freqs):
    h22 = get_h22(params, freqs)
    return h22.amp * np.exp(1.j * h22.phase)


# define inner product between waveforms in frequency-domain
# uses scipy's rhombus numerical integration
def inner(a, b, Ss, df):
    integrand = 4. * a.conjugate() * b / Ss
    inner_prod = integrate(integrand, dx=df)
    return inner_prod


step_sizes = [1e-1, 1e-1, 1e-1, 1e-1, 1e-4]
# calculate partial derivative of frequency-domain waveform
def partial_waveform(params, freqs, index):
    dstep = np.zeros(num_params)
    dstep[index] = step_sizes[index]
    waveform1 = get_waveform_freq(params - dstep, freqs)
    waveform2 = get_waveform_freq(params + dstep, freqs)
    return (waveform2 - waveform1) / (2 * step_sizes[index])


# calculate match between waveforms
def match(params1, params2, freqs, Ss, df):
    waveform1 = get_waveform_freq(params1, freqs)
    waveform2 = get_waveform_freq(params2, freqs)
    SNRsq1 = inner(waveform1, waveform1, Ss, df)
    SNRsq2 = inner(waveform2, waveform2, Ss, df)
    cross = inner(waveform1, waveform2, Ss, df)
    return cross / np.sqrt(SNRsq1 * SNRsq2)


# get components of metric on template space
def metric(params, freqs, Ss, df):
    waveform = get_waveform_freq(params, freqs)
    partials = [partial_waveform(params, freqs, i) for i in range(num_params)]
    SNRsq = inner(waveform, waveform, Ss, df)
    SNR = np.sqrt(SNRsq)
    metric_comp = np.zeros((num_params, num_params))
    for i in range(num_params):
        for j in range(i, num_params):
            first_term = inner(partials[i], partials[j], Ss, df) / SNR
            second_term = inner(waveform, partials[i], Ss, df) * inner(waveform, partials[j], Ss, df) / SNRsq
            metric_comp[i,j] = metric_comp[j,i] = first_term - second_term
    return metric_comp


# project metric 








# load data
times = np.loadtxt('times.txt')
data_H1 = np.loadtxt('data_H1.txt')
data_L1 = np.loadtxt('data_L1.txt')
# get psd interpolation
[psd_H1, psd_L1] = get_psd(times, data_H1, data_L1)
def psd(f):
    return (1/psd_H1(f) + 1/psd_L1(f))**(-1)
# define frequency bins
fs = np.linspace(20, 2048, 2**9 + 1)
df = fs[1] - fs[0]


params = [m1_measured, m2_measured, 0., 0., 100.]
metric_comp = metric(params, fs, psd(fs), df)
print(metric_comp)





